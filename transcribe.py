import logging
import os
import re
import tempfile
from typing import List, Literal, Union

import librosa
import numpy as np
import torch
from funasr_onnx import Fsmn_vad_online, SenseVoiceSmall
from pysrt import SubRipFile, SubRipItem, SubRipTime
from resampy.core import resample
from torchaudio.pipelines import MMS_FA as bundle
from tqdm.auto import tqdm

from corrector import BertModel, corrector
from denoiser import denoiser
from utils import load_dict

logger = logging.getLogger(__name__)

asr_model = SenseVoiceSmall("./models/iic/SenseVoiceSmall", batch_size=1, quantize=True)
vad_model = Fsmn_vad_online(
    "./models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", batch_size=1, quantize=True
)

LABELS = [
    asr_model.tokenizer.sp.IdToPiece(i)
    for i in range(asr_model.tokenizer.sp.piece_size())
]
special_labels = [
    label for label in LABELS if label.startswith("<|") and label.endswith("|>")
]

special_token_ids = [
    asr_model.tokenizer.sp.PieceToId(i)
    for i in ["<s>", "</s>", "<unk>", "<pad>"] + special_labels
]
aligner = bundle.get_aligner()
SAMPLE_RATE = 16000

t2s_char_dict, char_jyutping_dict, jyutping_char_dict, chars_freq = load_dict()


class TranscribeResult:
    """
    Each TranscribeResult object represents one SRT line.
    """

    def __init__(self, text: str, start_time: float, end_time: float):
        self.text = text
        self.start_time = start_time
        self.end_time = end_time

    def __str__(self):
        return f"TranscribeResult(text={self.text}, start_time={self.start_time}, end_time={self.end_time})"

    def __repr__(self):
        return str(self)


def asr(
    wav_content: Union[str, np.ndarray, List[str]],
    language="yue",
    textnorm: Union[Literal["withitn", "woitn"]] = "withitn",
    with_punct=False,
) -> List["TranscribeResult"]:
    """
    Perform ASR on the given audio content. Returns a list of TranscribeResult objects which represent SRT lines.

    Args:
        wav_content (Union[str, np.ndarray, List[str]]): Audio content to transcribe.
        language (str, optional): Language code. Defaults to "yue".
        textnorm (Union[Literal["withitn", "woitn"]], optional): Text normalization. Defaults to "withitn".
        with_punct (bool, optional): Include punctuation. Defaults to False.

    Returns:
        List[TranscribeResult]: A list of TranscribeResult objects.
    """
    language_input = language
    textnorm_input = textnorm
    language_list, textnorm_list = asr_model.read_tags(language_input, textnorm_input)
    waveform = asr_model.load_data(
        wav_content, asr_model.frontend.opts.frame_opts.samp_freq
    )
    feats, feats_len = asr_model.extract_feat(waveform)
    _language_list = language_list[0:1]
    _textnorm_list = textnorm_list[0:1]
    segments = []
    results = []

    ctc_logits, _ = asr_model.infer(
        feats,
        feats_len,
        np.array(_language_list, dtype=np.int32),
        np.array(_textnorm_list, dtype=np.int32),
    )
    ctc_logits = torch.from_numpy(ctc_logits).float()
    ratio = waveform[0].shape[0] / ctc_logits.size(1) / SAMPLE_RATE
    # support batch_size=1 only currently
    x = ctc_logits[0]
    # log_softmax
    x = torch.nn.functional.log_softmax(x, dim=-1)
    yseq = x.argmax(dim=-1)
    yseq = torch.unique_consecutive(yseq, dim=-1)

    mask = yseq != asr_model.blank_id
    preds = yseq[mask]
    token_spans = aligner(ctc_logits[0], preds.unsqueeze(0))[0]
    results = _process_segments(token_spans, ratio, with_punct)

    return results


def _process_segments(
    token_spans,
    ratio,
    with_punct,
    punct_labels=",?!。，；？！",
    sep_labels="?!。；？！",
):
    """
    Processes token spans into segments with optional punctuation handling.

    Args:
        token_spans (list): A list of token spans, where each span contains a token, start time, and end time.
        ratio (float): A ratio to adjust the start and end times of the segments.
        with_punct (bool): A flag indicating whether to include punctuation in the segments.
        punct_labels (str, optional): A string of punctuation characters to consider. Defaults to "?!。，；？！".
        sep_labels (str, optional): A string of separator characters to consider. Defaults to "?!。；？

    Returns:
        list: A list of TranscribeResult objects, each containing text, start time, and end time.
    """
    segments = []

    for token_span in token_spans:
        label = asr_model.tokenizer.sp.IdToPiece(token_span.token)

        if token_span.token in special_token_ids:
            continue

        segments.append(
            TranscribeResult(
                text=label,
                start_time=token_span.start * ratio,
                end_time=token_span.end * ratio,
            )
        )

    results = []
    start_idx = 0
    end_idx = 1

    while end_idx < len(segments):
        current_segment = segments[end_idx]

        if current_segment.text in sep_labels:
            curr_end_idx = end_idx

            if with_punct:
                curr_end_idx += 1

            text = "".join(
                [segment.text for segment in segments[start_idx:curr_end_idx]]
            )

            if not with_punct:
                text = re.sub(f"[{punct_labels}]", " ", text)

            results.append(
                TranscribeResult(
                    text=text,
                    start_time=segments[start_idx].start_time,
                    end_time=segments[end_idx].end_time,
                )
            )
            start_idx = end_idx + 1

        end_idx += 1

    return results


def slice_padding_audio_samples(speech, speech_lengths, vad_segments):
    """
    Slice and pad audio samples based on VAD segments.
    """
    speech_list = []
    speech_lengths_list = []
    for _, segment in enumerate(vad_segments):
        bed_idx = int(segment[0][0] * 16)
        end_idx = min(int(segment[0][1] * 16), speech_lengths)
        speech_i = speech[bed_idx:end_idx]
        speech_lengths_i = end_idx - bed_idx
        speech_list.append(speech_i)
        speech_lengths_list.append(speech_lengths_i)

    return speech_list, speech_lengths_list


bert_model = BertModel("./models/hon9kon9ize/bert-large-cantonese")


def transcribe(audio_file: str) -> List["TranscribeResult"]:
    speech, sr = librosa.load(audio_file)
    speech, new_sr = denoiser(speech, sr)

    if new_sr != 16_000:
        speech = resample(speech, new_sr, 16_000, filter="kaiser_best", parallel=True)

    logger.info("Segmenting speech")

    res = vad_model(speech)
    vadsegments = res[0]
    n = len(vadsegments)
    speech_lengths = len(speech)
    results = []

    logger.info("Number of segments: %d", n)

    if not n:
        return []

    for j, _ in tqdm(enumerate(range(n)), total=n, desc="Transcribing"):
        speech_j, speech_lengths_j = slice_padding_audio_samples(
            speech, speech_lengths, [[vadsegments[j]]]
        )
        stt_results = asr(speech_j[0])
        timestamp_offset = ((vadsegments[j][0] * 16) / 16_000) - 0.1

        if len(stt_results) < 1:
            continue

        # add timestamp offset
        for result in stt_results:
            result.start_time += timestamp_offset
            result.end_time += timestamp_offset

        results.extend(stt_results)

    # convert to Traditional Chinese
    for result in tqdm(
        results, total=len(results), desc="Converting to Traditional Chinese"
    ):
        result.text = corrector(result.text, t2s_char_dict, bert_model)

    return results


def to_srt(results: List["TranscribeResult"]) -> str:
    """
    Convert the list of TranscrbeResult objects into a SRT file
    """
    srt = SubRipFile()

    for i, t in enumerate(results):
        start = SubRipTime(seconds=t.start_time)
        end = SubRipTime(seconds=t.end_time)
        item = SubRipItem(index=i, start=start, end=end, text=t.text)
        srt.append(item)

    temp_file = tempfile.gettempdir() + "/output.srt"
    srt.save(temp_file)

    with open(temp_file, "r", encoding="utf-8") as f:
        srt_text = f.read()

    os.remove(temp_file)

    return srt_text
