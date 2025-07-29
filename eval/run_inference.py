import argparse
import json
import os
import torch
import unicodedata
import re
from dataclasses import dataclass
from datasets import load_dataset
from tqdm.auto import tqdm
from transcriber.AutoTranscriber import AutoTranscriber
from typing import Dict, List

# Config
DATASET_CONFIGS = {
    "ming030890/cantonese_asr_eval_mdcc_long": {
        "audio_key": "audio",
        "text_key": "transcript",
        "split": "test",
        "sample_size": 1000,
    },
    "JackyHoCL/common_voice_22_yue": {
        "audio_key": "audio",
        "text_key": "sentence",
        "split": "test",
        "sample_size": 1000,
    },
}

def run_asr_eval(transcriber, dataset_name, jsonl_writer):
    cfg = DATASET_CONFIGS[dataset_name]
    ds = (
        load_dataset(dataset_name, split=cfg["split"], streaming=True)
        .shuffle(buffer_size=cfg["sample_size"] * 10, seed=42)
        .take(cfg["sample_size"])
    )

    with tqdm(
        ds,
        desc=f"[{dataset_name.split('/')[-1]}]",
        total=cfg["sample_size"],
        unit="utt",
        leave=False,
    ) as pbar:
        for row in pbar:
            audio = row[cfg["audio_key"]]["array"]
            text = row[cfg["text_key"]]
            sample_rate = row[cfg["audio_key"]]["sampling_rate"]

            with torch.no_grad():
                results = transcriber.transcribe_audio_bytes(audio, sample_rate)
            prediction = "".join([res.text for res in results])

            jsonl_writer.write(
                json.dumps(
                    {"dataset": dataset_name, "reference": text, "prediction": prediction},
                    ensure_ascii=False,
                )
                + "\n"
            )

def main():
    parser = argparse.ArgumentParser(description="Run ASR evaluation with configurable corrector")
    parser.add_argument(
        "--corrector",
        type=str,
        default="opencc",
        help="Text corrector to apply (e.g., opencc, none)",
    )
    parser.add_argument(
        "--use_denoiser",
        action="store_true",
        help="Whether to enable denoiser",
    )
    parser.add_argument(
        "--with_punct",
        action="store_true",
        help="Whether to include punctuation in the output",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save the output JSONL file",
    )
    args = parser.parse_args()

    # Construct flags for filename based on parameters
    denoiser_flag = "denoiser" if args.use_denoiser else "no_denoiser"
    punct_flag = "punct" if args.with_punct else "no_punct"
    # Prepare output file path reflecting all params
    filename = f"asr_results_corrector_{args.corrector}_{denoiser_flag}_{punct_flag}.jsonl"
    output_path = os.path.join(args.output_dir, filename)

    with open(f'output/{output_path}', "w", encoding="utf-8") as jsonl_file:
        transcriber = AutoTranscriber(
            corrector=args.corrector,
            use_denoiser=args.use_denoiser,
            with_punct=args.with_punct,
        )

        with tqdm(
            DATASET_CONFIGS.keys(),
            desc="Datasets",
            total=len(DATASET_CONFIGS),
            unit="ds",
        ) as ds_bar:
            for ds_name in ds_bar:
                run_asr_eval(transcriber, ds_name, jsonl_file)
                # force bar refresh
                ds_bar.update(0)

    print(f"Results written to {output_path}")
    # Workaround for https://github.com/huggingface/datasets/issues/7467
    os._exit(0)

if __name__ == "__main__":
    main()
