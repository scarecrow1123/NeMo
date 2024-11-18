# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script can be used to simulate cache-aware streaming for ASR models. The ASR model to be used with this script need to get trained in streaming mode. Currently only Conformer models supports this streaming mode.
You may find examples of streaming models under 'NeMo/example/asr/conf/conformer/streaming/'.

It works both on a manifest of audio files or a single audio file. It can perform streaming for a single stream (audio) or perform the evalution in multi-stream model (batch_size>1).
The manifest file must conform to standard ASR definition - containing `audio_filepath` and `text` as the ground truth.

# Usage

## To evaluate a model in cache-aware streaming mode on a single audio file:

python speech_to_text_streaming_infer.py \
    --asr_model=asr_model.nemo \
    --audio_file=audio_file.wav \
    --compare_vs_offline \
    --use_amp \
    --debug_mode

## To evaluate a model in cache-aware streaming mode on a manifest file:

python speech_to_text_streaming_infer.py \
    --asr_model=asr_model.nemo \
    --manifest_file=manifest_file.json \
    --batch_size=16 \
    --compare_vs_offline \
    --use_amp \
    --debug_mode

You may drop the '--debug_mode' and '--compare_vs_offline' to speedup the streaming evaluation.
If compare_vs_offline is not used, then significantly larger batch_size can be used.
Setting `--pad_and_drop_preencoded` would perform the caching for all steps including the first step.
It may result in slightly different outputs from the sub-sampling module compared to offline mode for some techniques like striding and sw_striding.
Enabling it would make it easier to export the model to ONNX.

## Hybrid ASR models
For Hybrid ASR models which have two decoders, you may select the decoder by --set_decoder DECODER_TYPE, where DECODER_TYPE can be "ctc" or "rnnt".
If decoder is not set, then the default decoder would be used which is the RNNT decoder for Hybrid ASR models.

## Multi-lookahead models
For models which support multiple lookaheads, the default is the first one in the list of model.encoder.att_context_size. To change it, you may use --att_context_size, for example --att_context_size [70,1].


## Evaluate a model trained with full context for offline mode

You may try the cache-aware streaming with a model trained with full context in offline mode.
But the accuracy would not be very good with small chunks as there is inconsistency between how the model is trained and how the streaming inference is done.
The accuracy of the model on the borders of chunks would not be very good.

To use a model trained with full context, you need to pass the chunk_size and shift_size arguments.
If shift_size is not passed, chunk_size would be used as the shift_size too.
Also argument online_normalization should be enabled to simulate a realistic streaming.
The following command would simulate cache-aware streaming on a pretrained model from NGC with chunk_size of 100, shift_size of 50 and 2 left chunks as left context.
The chunk_size of 100 would be 100*4*10=4000ms for a model with 4x downsampling and 10ms shift in feature extraction.

python speech_to_text_streaming_infer.py \
    --asr_model=stt_en_conformer_ctc_large \
    --chunk_size=100 \
    --shift_size=50 \
    --left_chunks=2 \
    --online_normalization \
    --manifest_file=manifest_file.json \
    --batch_size=16 \
    --compare_vs_offline \
    --use_amp \
    --debug_mode

"""


import contextlib
import json
import os
import time
from argparse import ArgumentParser
from typing import List

import torch
from omegaconf import open_dict

import onnxruntime as ort

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
from nemo.utils import logging


def extract_transcriptions(hyps):
    """
    The transcribed_texts returned by CTC and RNNT models are different.
    This method would extract and return the text section of the hypothesis.
    """
    if isinstance(hyps[0], Hypothesis):
        transcriptions = []
        for hyp in hyps:
            transcriptions.append(hyp.text)
    else:
        transcriptions = hyps
    return transcriptions


def calc_drop_extra_pre_encoded(asr_model, step_num, pad_and_drop_preencoded):
    # for the first step there is no need to drop any tokens after the downsampling as no caching is being used
    if step_num == 0 and not pad_and_drop_preencoded:
        return 0
    else:
        return asr_model.encoder.streaming_cfg.drop_extra_pre_encoded


def onnx_stream_step(
    onnx_model,
    nemo_model,
    device,
    processed_signal: torch.Tensor,
    processed_signal_length: torch.Tensor = None,
    cache_last_channel: torch.Tensor = None,
    cache_last_time: torch.Tensor = None,
    cache_last_channel_len: torch.Tensor = None,
    keep_all_outputs: bool = True,
    previous_hypotheses: List[Hypothesis] = None,
    previous_pred_out: torch.Tensor = None,
    drop_extra_pre_encoded: int = None,
    return_transcription: bool = True,
    return_log_probs: bool = False,
):
    # print(f"cache lc: {cache_last_channel.transpose(1, 0).shape} | cache lt: {cache_last_time.transpose(0, 1).shape} | cache lc len: {cache_last_channel_len.shape}")
    (
        encoded,
        encoded_len,
        cache_last_channel_next,
        cache_last_time_next,
        cache_last_channel_next_len,
    ) = onnx_model.run(
        [i.name for i in onnx_model.get_outputs()],
        {
            "audio_signal": processed_signal.cpu().numpy(),
            "length": processed_signal_length.cpu().numpy(),
            "cache_last_channel": cache_last_channel.transpose(0, 1).cpu().numpy(),
            "cache_last_time": cache_last_time.transpose(0, 1).cpu().numpy(),
            "cache_last_channel_len": cache_last_channel_len.cpu().numpy()
        }
    )

    encoded = torch.from_numpy(encoded).to(device=device)
    encoded_len = torch.from_numpy(encoded_len).to(device=device)
    cache_last_channel_next = torch.from_numpy(cache_last_channel_next).to(device=device).transpose(1, 0)
    cache_last_time_next = torch.from_numpy(cache_last_time_next).to(device=device).transpose(1, 0)
    cache_last_channel_next_len = torch.from_numpy(cache_last_channel_next_len).to(device=device)

    # print(f"encoded: {encoded.shape} | encoded_len: {encoded_len.shape}")
    # print(f"cache last c next: {cache_last_channel_next.shape} | cache last t next: {cache_last_time_next.shape}")

    decoding = nemo_model.ctc_decoding
    decoder = nemo_model.ctc_decoder

    log_probs = encoded
    # log_probs = decoder(encoder_output=encoded)
    predictions_tensor = log_probs.argmax(dim=-1, keepdim=False)

    # Concatenate the previous predictions with the current one to have the full predictions.
    # We drop the extra predictions for each sample by using the lengths returned by the encoder (encoded_len)
    # Then create a list of the predictions for the batch. The predictions can have different lengths because of the paddings.
    greedy_predictions = []
    if return_transcription:
        all_hyp_or_transcribed_texts = []
    else:
        all_hyp_or_transcribed_texts = None

    for preds_idx, preds in enumerate(predictions_tensor):
        if encoded_len is None:
            preds_cur = predictions_tensor[preds_idx]
        else:
            preds_cur = predictions_tensor[preds_idx, : encoded_len[preds_idx]]
        if previous_pred_out is not None:
            greedy_predictions_concat = torch.cat((previous_pred_out[preds_idx], preds_cur), dim=-1)
            encoded_len[preds_idx] += len(previous_pred_out[preds_idx])
        else:
            greedy_predictions_concat = preds_cur
        greedy_predictions.append(greedy_predictions_concat)

        # TODO: make decoding more efficient by avoiding the decoding process from the beginning
        if return_transcription:
            decoded_out = decoding.ctc_decoder_predictions_tensor(
                decoder_outputs=greedy_predictions_concat.unsqueeze(0),
                decoder_lengths=encoded_len[preds_idx : preds_idx + 1],
                return_hypotheses=False,
            )
            all_hyp_or_transcribed_texts.append(decoded_out[0][0])
    best_hyp = None

    result = [
        greedy_predictions,
        all_hyp_or_transcribed_texts,
        cache_last_channel_next,
        cache_last_time_next,
        cache_last_channel_next_len,
        best_hyp,
    ]
    if return_log_probs:
        result.append(log_probs)
        result.append(encoded_len)

    return tuple(result)

def perform_onnx_streaming(
        onnx_model, nemo_model, device, streaming_buffer, compare_vs_offline=False, debug_mode=False, pad_and_drop_preencoded=False
):
    batch_size = len(streaming_buffer.streams_length)
    cache_last_channel, cache_last_time, cache_last_channel_len = nemo_model.encoder.get_initial_cache_state(
        batch_size=batch_size
    )

    previous_hypotheses = None
    streaming_buffer_iter = iter(streaming_buffer)
    pred_out_stream = None
    for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer_iter):
        (
            pred_out_stream,
            transcribed_texts,
            cache_last_channel,
            cache_last_time,
            cache_last_channel_len,
            previous_hypotheses,
        ) = onnx_stream_step(
            onnx_model,
            nemo_model,
            device,
            processed_signal=chunk_audio,
            processed_signal_length=chunk_lengths,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=streaming_buffer.is_buffer_empty(),
            previous_hypotheses=previous_hypotheses,
            previous_pred_out=pred_out_stream,
            drop_extra_pre_encoded=calc_drop_extra_pre_encoded(
                nemo_model, step_num, pad_and_drop_preencoded
            ),
            return_transcription=True,
        )
    
    final_streaming_tran = extract_transcriptions(transcribed_texts)
    logging.info(f"Final streaming transcriptions: {final_streaming_tran}")

    return final_streaming_tran

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--asr_model",
        type=str,
        required=True,
        help="Path to an ASR model .nemo file or name of a pretrained model.",
    )
    parser.add_argument(
        "--device", type=str, help="The device to load the model onto and perform the streaming", default="cuda"
    )
    parser.add_argument("--audio_file", type=str, help="Path to an audio file to perform streaming", default=None)
    parser.add_argument(
        "--manifest_file",
        type=str,
        help="Path to a manifest file containing audio files to perform streaming",
        default=None,
    )
    parser.add_argument("--use_amp", action="store_true", help="Whether to use AMP")
    parser.add_argument("--debug_mode", action="store_true", help="Whether to print more detail in the output.")
    parser.add_argument(
        "--compare_vs_offline",
        action="store_true",
        help="Whether to compare the output of the model with the offline mode.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="The batch size to be used to perform streaming in batch mode with multiple streams",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=-1,
        help="The chunk_size to be used for models trained with full context and offline models",
    )
    parser.add_argument(
        "--shift_size",
        type=int,
        default=-1,
        help="The shift_size to be used for models trained with full context and offline models",
    )
    parser.add_argument(
        "--left_chunks",
        type=int,
        default=2,
        help="The number of left chunks to be used as left context via caching for offline models",
    )

    parser.add_argument(
        "--online_normalization",
        default=False,
        action='store_true',
        help="Perform normalization on the run per chunk.",
    )
    parser.add_argument(
        "--output_path", type=str, help="path to output file when manifest is used as input", default=None
    )
    parser.add_argument(
        "--pad_and_drop_preencoded",
        action="store_true",
        help="Enables padding the audio input and then dropping the extra steps after the pre-encoding for all the steps including the the first step. It may make the outputs of the downsampling slightly different from offline mode for some techniques like striding or sw_striding.",
    )

    parser.add_argument(
        "--set_decoder",
        choices=["ctc", "rnnt"],
        default=None,
        help="Selects the decoder for Hybrid ASR models which has both the CTC and RNNT decoder. Supported decoders are ['ctc', 'rnnt']",
    )

    parser.add_argument(
        "--att_context_size",
        type=str,
        default=None,
        help="Sets the att_context_size for the models which support multiple lookaheads",
    )

    args = parser.parse_args()
    if (args.audio_file is None and args.manifest_file is None) or (
        args.audio_file is not None and args.manifest_file is not None
    ):
        raise ValueError("One of the audio_file and manifest_file should be non-empty!")

    if args.asr_model.endswith('.nemo'):
        logging.info(f"Using local ASR model from {args.asr_model}")
        asr_model = nemo_asr.models.ASRModel.restore_from(restore_path=args.asr_model)
    else:
        logging.info(f"Using NGC cloud ASR model {args.asr_model}")
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=args.asr_model)

    logging.info(asr_model.encoder.streaming_cfg)
    if args.set_decoder is not None:
        if hasattr(asr_model, "cur_decoder"):
            asr_model.change_decoding_strategy(decoder_type=args.set_decoder)
        else:
            raise ValueError("Decoder cannot get changed for non-Hybrid ASR models.")

    if args.att_context_size is not None:
        if hasattr(asr_model.encoder, "set_default_att_context_size"):
            asr_model.encoder.set_default_att_context_size(att_context_size=json.loads(args.att_context_size))
        else:
            raise ValueError("Model does not support multiple lookaheads.")

    global autocast
    autocast = torch.amp.autocast(asr_model.device.type, enabled=args.use_amp)

    # configure the decoding config
    decoding_cfg = asr_model.cfg.decoding
    with open_dict(decoding_cfg):
        decoding_cfg.strategy = "greedy"
        decoding_cfg.preserve_alignments = False
        if hasattr(asr_model, 'joint'):  # if an RNNT model
            decoding_cfg.greedy.max_symbols = 10
            decoding_cfg.fused_batch_size = -1
        asr_model.change_decoding_strategy(decoding_cfg)

    asr_model = asr_model.to(args.device)
    asr_model.eval()

    # chunk_size is set automatically for models trained for streaming. For models trained for offline mode with full context, we need to pass the chunk_size explicitly.
    if args.chunk_size > 0:
        if args.shift_size < 0:
            shift_size = args.chunk_size
        else:
            shift_size = args.shift_size
        asr_model.encoder.setup_streaming_params(
            chunk_size=args.chunk_size, left_chunks=args.left_chunks, shift_size=shift_size
        )

    # In streaming, offline normalization is not feasible as we don't have access to the whole audio at the beginning
    # When online_normalization is enabled, the normalization of the input features (mel-spectrograms) are done per step
    # It is suggested to train the streaming models without any normalization in the input features.
    if args.online_normalization:
        if asr_model.cfg.preprocessor.normalize not in ["per_feature", "all_feature"]:
            logging.warning(
                "online_normalization is enabled but the model has no normalization in the feature extration part, so it is ignored."
            )
            online_normalization = False
        else:
            online_normalization = True

    else:
        online_normalization = False

    onnx_model = ort.InferenceSession("/home/seelan/workspace/models/streaming/temp/artifacts/model_graph.onnx")#, providers=["CUDAExecutionProvider"])

    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=asr_model,
        online_normalization=online_normalization,
        pad_and_drop_preencoded=args.pad_and_drop_preencoded,
    )
    if args.audio_file is not None:
        # stream a single audio file
        processed_signal, processed_signal_length, stream_id = streaming_buffer.append_audio_file(
            args.audio_file, stream_id=-1
        )
        perform_onnx_streaming(
            onnx_model,
            asr_model,
            args.device,
            streaming_buffer=streaming_buffer,
            compare_vs_offline=args.compare_vs_offline,
            pad_and_drop_preencoded=args.pad_and_drop_preencoded,
        )
    else:
        # stream audio files in a manifest file in batched mode
        samples = []
        all_streaming_tran = []
        all_offline_tran = []
        all_refs_text = []

        with open(args.manifest_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                samples.append(item)

        logging.info(f"Loaded {len(samples)} from the manifest at {args.manifest_file}.")

        start_time = time.time()
        for sample_idx, sample in enumerate(samples):
            processed_signal, processed_signal_length, stream_id = streaming_buffer.append_audio_file(
                sample['audio_filepath'], stream_id=-1
            )
            if "text" in sample:
                all_refs_text.append(sample["text"])
            logging.info(f'Added this sample to the buffer: {sample["audio_filepath"]}')

            if (sample_idx + 1) % args.batch_size == 0 or sample_idx == len(samples) - 1:
                logging.info(f"Starting to stream samples {sample_idx - len(streaming_buffer) + 1} to {sample_idx}...")
                streaming_tran = perform_onnx_streaming(
                    onnx_model,
                    asr_model,
                    args.device,
                    streaming_buffer=streaming_buffer,
                    compare_vs_offline=args.compare_vs_offline,
                    debug_mode=args.debug_mode,
                    pad_and_drop_preencoded=args.pad_and_drop_preencoded,
                )
                all_streaming_tran.extend(streaming_tran)
                if args.compare_vs_offline:
                    all_offline_tran.extend(offline_tran)
                streaming_buffer.reset_buffer()

        if args.compare_vs_offline and len(all_refs_text) == len(all_offline_tran):
            offline_wer = word_error_rate(hypotheses=all_offline_tran, references=all_refs_text)
            logging.info(f"WER% of offline mode: {round(offline_wer * 100, 2)}")
        if len(all_refs_text) == len(all_streaming_tran):
            streaming_wer = word_error_rate(hypotheses=all_streaming_tran, references=all_refs_text)
            logging.info(f"WER% of streaming mode: {round(streaming_wer*100, 2)}")

        end_time = time.time()
        logging.info(f"The whole streaming process took: {round(end_time - start_time, 2)}s")

        # stores the results including the transcriptions of the streaming inference in a json file
        if args.output_path is not None and len(all_refs_text) == len(all_streaming_tran):
            fname = (
                "streaming_out_"
                + os.path.splitext(os.path.basename(args.asr_model))[0]
                + "_"
                + os.path.splitext(os.path.basename(args.manifest_file))[0]
                + ".json"
            )

            hyp_json = os.path.join(args.output_path, fname)
            os.makedirs(args.output_path, exist_ok=True)
            with open(hyp_json, "w") as out_f:
                for i, hyp in enumerate(all_streaming_tran):
                    record = {
                        "pred_text": hyp,
                        "text": all_refs_text[i],
                        "wer": round(word_error_rate(hypotheses=[hyp], references=[all_refs_text[i]]) * 100, 2),
                    }
                    out_f.write(json.dumps(record) + '\n')


if __name__ == '__main__':
    main()
