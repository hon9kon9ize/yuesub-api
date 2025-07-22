import logging
import torch
import time
from typing import List, Literal

import librosa
import numpy as np
from funasr import AutoModel
from resampy.core import resample
from tqdm.auto import tqdm

from denoiser import denoiser
from transcriber.Transcriber import Transcriber, TranscribeResult

logger = logging.getLogger(__name__)


class AutoTranscriber(Transcriber):
    """
    Transcriber class that uses FunASR's AutoModel for VAD and ASR
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Initialize models
        self.vad_model = AutoModel(
            model="fsmn-vad",
            max_single_segment_time=self.max_length_seconds * 1000,
        )
        self.asr_model = AutoModel(
            model="iic/SenseVoiceSmall",
            vad_model=None,  # We'll handle VAD separately
            punc_model=None,
            ban_emo_unks=True,
            device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        )

    def transcribe(
        self,
        audio_file: str,
    ) -> List[TranscribeResult]:
        """
        Transcribe audio file to text with timestamps.

        Args:
            audio_file (str): Path to audio file

        Returns:
            List[TranscribeResult]: List of transcription results
        """
        # Load and preprocess audio
        speech, sr = librosa.load(audio_file, sr=self.sr)

        if self.use_denoiser:
            logger.info("Denoising speech...")
            speech, _ = denoiser(speech, sr)

        if sr != 16_000:
            speech = resample(speech, sr, 16_000,
                              filter="kaiser_best", parallel=True)

        # Get VAD segments
        logger.info("Segmenting speech...")

        start_time = time.time()
        vad_results = self.vad_model.generate(input=speech, disable_pbar=True)
        logger.info("VAD took %.2f seconds", time.time() - start_time)

        if not vad_results or not vad_results[0]["value"]:
            return []

        vad_segments = vad_results[0]["value"]

        # Process each segment
        results = []

        start_time = time.time()
        for segment in tqdm(vad_segments, desc="Transcribing"):
            start_sample = int(segment[0] * 16)  # Convert ms to samples
            end_sample = int(segment[1] * 16)
            segment_audio = speech[start_sample:end_sample]

            # Get ASR results for segment
            asr_result = self.asr_model.generate(
                input=segment_audio, language="yue", use_itn=self.with_punct, disable_pbar=True
            )

            if not asr_result:
                continue

            start_segment_time = max(0, segment[0] / 1000.0 + self.offset_in_seconds)
            end_segment_time = segment[1] / 1000.0 + self.offset_in_seconds

            # Convert ASR result to TranscribeResult format
            segment_result = TranscribeResult(
                text=asr_result[0]["text"],
                start_time=start_segment_time,  # Convert ms to seconds
                end_time=end_segment_time,
            )
            results.append(segment_result)

        logger.info("ASR took %.2f seconds", time.time() - start_time)

        # Apply Chinese conversion if needed
        start_time = time.time()
        results = self._postprocessing(results)
        logger.info("Conversion took %.2f seconds", time.time() - start_time)

        return results
