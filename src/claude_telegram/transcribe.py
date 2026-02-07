"""Audio transcription — Whisper local + Voxtral API fallback."""

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import httpx

from .config import settings

logger = logging.getLogger(__name__)

DURATION_THRESHOLD = 300  # 5 minutes — above this, use Voxtral


@dataclass
class TranscriptionResult:
    text: str
    engine: str
    duration: float
    duration_formatted: str


def get_audio_duration(file_path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", file_path],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return float(result.stdout.strip())


def convert_to_wav(input_path: str) -> str:
    """Convert audio to WAV 16kHz mono (required by whisper.cpp)."""
    wav_path = str(Path(input_path).with_suffix(".wav"))
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")
    return wav_path


def transcribe_whisper(wav_path: str) -> TranscriptionResult:
    """Transcribe using local whisper.cpp."""
    result = subprocess.run(
        [settings.whisper_bin, "-m", settings.whisper_model, "-f", wav_path,
         "-l", "fr", "--no-timestamps", "-t", "4", "-np"],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Whisper failed: {result.stderr}")

    duration = get_audio_duration(wav_path)
    return TranscriptionResult(
        text=result.stdout.strip(),
        engine="whisper-medium-local",
        duration=duration,
        duration_formatted=f"{duration / 60:.1f} min",
    )


async def transcribe_voxtral(audio_path: str) -> TranscriptionResult:
    """Transcribe using Voxtral API (Mistral)."""
    if not settings.mistral_api_key:
        raise RuntimeError("MISTRAL_API_KEY not set")

    duration = get_audio_duration(audio_path)

    async with httpx.AsyncClient(timeout=120) as client:
        with open(audio_path, "rb") as f:
            response = await client.post(
                "https://api.mistral.ai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {settings.mistral_api_key}"},
                files={"file": (Path(audio_path).name, f)},
                data={"model": "voxtral-mini-transcribe-2602", "language": "fr"},
            )
        if response.status_code != 200:
            raise RuntimeError(f"Voxtral API error {response.status_code}: {response.text}")

        data = response.json()

    return TranscriptionResult(
        text=data["text"],
        engine="voxtral-mini-transcribe-v2",
        duration=duration,
        duration_formatted=f"{duration / 60:.1f} min",
    )


async def transcribe_audio(audio_path: str) -> TranscriptionResult:
    """Transcribe audio — pick engine based on duration."""
    wav_path = convert_to_wav(audio_path)
    try:
        duration = get_audio_duration(wav_path)
        if duration < DURATION_THRESHOLD:
            return transcribe_whisper(wav_path)
        else:
            return await transcribe_voxtral(audio_path)
    finally:
        try:
            Path(wav_path).unlink(missing_ok=True)
        except Exception:
            pass
