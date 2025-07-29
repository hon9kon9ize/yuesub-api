import json
from dataclasses import dataclass
from datasets import load_dataset
from jiwer import cer
from opencc import OpenCC
from tqdm.auto import tqdm
from transcriber.AutoTranscriber import AutoTranscriber
from typing import Dict, List
import os
import torch
import re
import unicodedata

# Config
DATASET_CONFIGS = {
    "ming030890/cantonese_asr_eval_mdcc_long": {
        "audio_key": "audio",
        "text_key": "transcript",
        "split": "test",
        "sample_size": 100,
    },
    "JackyHoCL/common_voice_22_yue": {
        "audio_key": "audio",
        "text_key": "sentence",
        "split": "test",
        "sample_size": 100,
    },
}

opencc = OpenCC("hk2s")


def remove_punctuation(text: str) -> str:
    return "".join(ch for ch in text if unicodedata.category(ch)[0] != "P")


def remove_spaces(text: str):
    return re.sub(r"\s+", "", text)


def post_process(text: str) -> str:
    text = opencc.convert(text)
    text = remove_punctuation(text)
    text = remove_spaces(text)
    return text


def run_asr_eval(transcriber, dataset_name, jsonl_writer):
    cfg = DATASET_CONFIGS[dataset_name]
    ds = (
        load_dataset(dataset_name, split=cfg["split"], streaming=True)
        .shuffle(buffer_size=cfg["sample_size"] * 10, seed=42)
        .take(cfg["sample_size"])
    )

    all_preds, all_labels = [], []

    # inner progress bar knows its total length
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

            ref = post_process(text)
            hyp = post_process(prediction)

            jsonl_writer.write(
                json.dumps(
                    {"dataset": dataset_name, "reference": ref, "hypothesis": hyp},
                    ensure_ascii=False,
                )
                + "\n"
            )

            all_labels.append(ref)
            all_preds.append(hyp)

            # update CER as a postfix every few utterances
            if len(all_preds) % 5 == 0:
                curr_cer = cer(all_labels, all_preds)
                pbar.set_postfix(cer=f"{curr_cer:.2%}")

    cer_score = cer(all_labels, all_preds)
    tqdm.write(f"ASR CER on {dataset_name}: {cer_score:.3%}")


if __name__ == "__main__":
    output_path = "asr_results.jsonl"
    with open(output_path, "w", encoding="utf-8") as jsonl_file:
        transcriber = AutoTranscriber(
            corrector="opencc", use_denoiser=False, with_punct=False
        )

        with tqdm(
            DATASET_CONFIGS.keys(),
            desc="Datasets",
            total=len(DATASET_CONFIGS),
            unit="ds",
        ) as ds_bar:
            for ds_name in ds_bar:
                run_asr_eval(transcriber, ds_name, jsonl_file)
                ds_bar.update(0)  # ensure bar stays correct if nested updates
    print(f"Results written to {output_path}")
    # To workaround https://github.com/huggingface/datasets/issues/7467
    os._exit(0)
