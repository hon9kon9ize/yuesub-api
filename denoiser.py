# copy from https://github.com/skeskinen/resemble-denoise-onnx-inference/blob/master/denoiser.py

import numpy as np
from librosa import stft, istft
import onnxruntime
import torch
import gc
import os
from resampy.core import resample

stft_hop_length = 420
win_length = n_fft = 4 * stft_hop_length

# Configure device and providers
available_providers = onnxruntime.get_available_providers()
DEVICE = "cuda" if "CUDAExecutionProvider" in available_providers else "cpu"
PROVIDERS = (
    ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available_providers
    else ["CPUExecutionProvider"]
)

opts = onnxruntime.SessionOptions()
# Enable parallel execution
opts.inter_op_num_threads = 4
opts.intra_op_num_threads = 4
opts.enable_mem_pattern = True
opts.enable_cpu_mem_arena = True

print(__file__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'denoiser.onnx')
MODEL_PATH = os.path.abspath(MODEL_PATH)

session = onnxruntime.InferenceSession(
    MODEL_PATH,
    providers=PROVIDERS,
    sess_options=opts,
)


def _stft(x):
    s = stft(
        x,
        window="hann",
        win_length=win_length,
        n_fft=n_fft,
        hop_length=stft_hop_length,
        center=True,
        pad_mode="reflect",
    )

    s = s[..., :-1]

    mag = np.abs(s)

    phi = np.angle(s)
    cos = np.cos(phi)
    sin = np.sin(phi)

    return mag, cos, sin


def _istft(mag: np.array, cos: np.array, sin: np.array):
    real = mag * cos
    imag = mag * sin

    s = real + imag * 1.0j
    s = np.pad(s, ((0, 0), (0, 0), (0, 1)), mode="edge")
    x = istft(
        s, window="hann", win_length=win_length, hop_length=stft_hop_length, n_fft=n_fft
    )
    return x


def model(onnx_session, wav: np.array) -> np.array:
    padded_wav = np.pad(wav, ((0, 0), (0, 441)))

    mag, cos, sin = _stft(padded_wav)  # (b nfft/2 t)

    # Ensure contiguous memory layout for better performance
    mag = np.ascontiguousarray(mag, dtype=np.float32)
    cos = np.ascontiguousarray(cos, dtype=np.float32)
    sin = np.ascontiguousarray(sin, dtype=np.float32)

    ort_inputs = {
        "mag": mag,
        "cos": cos,
        "sin": sin,
    }

    sep_mag, sep_cos, sep_sin = onnx_session.run(None, ort_inputs)

    o = _istft(sep_mag, sep_cos, sep_sin)

    o = o[: wav.shape[-1]]
    return o


def run(
    onnx_session, wav: np.array, sample_rate: int, batch_process_chunks=False
) -> np.array:
    assert wav.ndim == 1, "Input should be 1D (mono) wav"

    if sample_rate != 44_100:
        wav = resample(wav, sample_rate, 44_100, filter="kaiser_best", parallel=True)

    chunk_length = int(sample_rate * 30)
    # overlap_length = int(sr * overlap_seconds)
    hop_length = chunk_length  # - overlap_length

    num_chunks = 1 + (wav.shape[-1] - 1) // hop_length
    n_pad = (num_chunks - wav.shape[-1] % num_chunks) % num_chunks
    wav = np.pad(wav, (0, n_pad))

    chunks = np.reshape(wav, (num_chunks, -1))
    abs_max = np.clip(
        np.max(np.abs(chunks), axis=-1, keepdims=True), a_min=1e-7, a_max=None
    )
    chunks /= abs_max

    if batch_process_chunks:
        res_chunks = model(onnx_session, chunks)
    else:
        res_chunks = np.array([model(onnx_session, c[None]) for c in chunks]).squeeze(
            axis=1
        )
    res_chunks *= abs_max

    res = np.reshape(res_chunks, (-1))
    return res[: wav.shape[-1]], 44_100


def denoiser(wav: np.array, sample_rate: int, batch_process_chunks=False) -> np.array:
    opts = onnxruntime.SessionOptions()
    session = onnxruntime.InferenceSession(
        "models/denoiser.onnx",
        providers=["CPUExecutionProvider"],
        # providers=["ROCMExecutionProvider"],
        # providers=["DnnlExecutionProvider"],
        sess_options=opts,
    )
    wav_onnx, new_sr = run(
        session, wav, sample_rate, batch_process_chunks=batch_process_chunks
    )
    del session
    gc.collect()

    return wav_onnx, new_sr
