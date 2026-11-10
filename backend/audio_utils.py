"""
Audio conversion: receives a cumulative WebM/Opus buffer (from the browser's
MediaRecorder) and returns 16kHz mono 16-bit PCM WAV.

We use PyAV (binary wheel with FFmpeg bundled) to avoid depending on a
system-wide ffmpeg in the Databricks App.
"""

from __future__ import annotations

import io
import logging

import av  # PyAV
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def webm_to_wav(audio_bytes: bytes, target_sr: int = 16000) -> bytes:
    """
    Converts any container supported by FFmpeg (WebM, Ogg, MP4, WAV,
    MP3...) to 16-bit mono PCM WAV at the requested sample rate.

    Strategy that is robust to partial MediaRecorder chunks: uses PyAV to
    decode all the audio, resamples to 16kHz mono, and serializes via
    soundfile.
    """
    if not audio_bytes:
        raise ValueError("audio_bytes is empty")

    # Decode with PyAV
    pcm_chunks: list[np.ndarray] = []
    try:
        container = av.open(io.BytesIO(audio_bytes))
    except av.error.InvalidDataError as exc:  # type: ignore[attr-defined]
        raise ValueError(f"PyAV could not open the stream: {exc}") from exc

    try:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise ValueError("Audio stream not found")

        resampler = av.AudioResampler(format="s16", layout="mono", rate=target_sr)
        for frame in container.decode(stream):
            for out_frame in resampler.resample(frame):
                arr = out_frame.to_ndarray()  # shape (channels, n) or (1, n)
                if arr.ndim == 2:
                    arr = arr[0]
                pcm_chunks.append(arr)
        # Flush the resampler
        for out_frame in resampler.resample(None):
            arr = out_frame.to_ndarray()
            if arr.ndim == 2:
                arr = arr[0]
            pcm_chunks.append(arr)
    finally:
        container.close()

    if not pcm_chunks:
        raise ValueError("No audio frames decoded")

    pcm = np.concatenate(pcm_chunks).astype(np.int16)
    logger.debug(
        "webm_to_wav: in=%d bytes -> out=%d samples @ %dHz",
        len(audio_bytes),
        len(pcm),
        target_sr,
    )

    # Serialize as WAV
    out = io.BytesIO()
    sf.write(out, pcm, target_sr, format="WAV", subtype="PCM_16")
    return out.getvalue()
