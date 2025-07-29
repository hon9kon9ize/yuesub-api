from tqdm.auto import tqdm
from corrector.Corrector import Corrector
import re
from typing import List, Literal


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


class Transcriber:
    def __init__(
        self,
        corrector: Literal["opencc", "bert", None] = None,
        use_denoiser=False,
        with_punct=True,
        offset_in_seconds=-0.25,
        max_length_seconds=10,
        sr=16000,
        quiet=False,
        batch_size=8,
    ):
        self.corrector = corrector
        self.use_denoiser = use_denoiser
        self.with_punct = with_punct
        self.sr = sr
        self.max_length_seconds = max_length_seconds
        self.offset_in_seconds = offset_in_seconds
        self.quiet = quiet
        self.batch_size = batch_size

    def transcribe(
        self,
        audio_file: str,
    ):
        raise NotImplementedError

    def transcribe_audio_bytes(
        self,
        audio_bytes: bytes,
    ):
        raise NotImplementedError

    def _postprocessing(
        self, results: List[TranscribeResult]
    ) -> List[TranscribeResult]:
        """Convert simplified Chinese to traditional Chinese"""
        if not results:
            return results

        if self.corrector is not None:
            corrector = Corrector(self.corrector)
            if self.corrector == "bert":
                for result in tqdm(
                    results,
                    total=len(results),
                    desc="Converting to Traditional Chinese",
                ):
                    result.text = corrector.correct(result.text)
            elif self.corrector == "opencc":
                # Use a special delimiter that won't appear in Chinese text
                delimiter = "|||"
                # Concatenate all texts with delimiter
                combined_text = delimiter.join(result.text for result in results)
                # Convert all text at once
                converted_text = corrector.correct(combined_text)
                # Split back into individual results
                converted_parts = converted_text.split(delimiter)

                # Update results with converted text
                for result, converted in zip(results, converted_parts):
                    result.text = converted

        return results
