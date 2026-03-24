"""Audio transcription — Whisper local + Voxtral API fallback + Sonnet post-correction."""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from .config import settings

logger = logging.getLogger(__name__)

# --- Context cache for post-correction (same as Omi STT proxy) ---

_CONTEXT_TTL = 300  # 5 minutes
_context_text: str = ""
_context_last_refresh: float = 0.0


def _read_context_file(path: Path, max_lines: int = 0) -> str:
    if not path.exists():
        return ""
    text = path.read_text()
    if max_lines > 0:
        lines = text.splitlines()[-max_lines:]
        return "\n".join(lines)
    return text


def _get_context() -> str:
    """Load GTD context for correction prompt, cached with TTL."""
    global _context_text, _context_last_refresh
    if time.time() - _context_last_refresh < _CONTEXT_TTL and _context_text:
        return _context_text

    ctx_dir = Path(os.environ.get("GTD_WORKING_DIR", "")) / "scripts" / "context"
    if not ctx_dir.exists():
        logger.warning(f"Context dir not found: {ctx_dir}")
        return ""

    profil = _read_context_file(ctx_dir / "profil.md")
    planning = _read_context_file(ctx_dir / "planning.md")
    faits = _read_context_file(ctx_dir / "faits-recents.md", max_lines=40)

    _context_text = f"""Contexte personnel de Thomas CHAUVET (pour orthographier correctement les noms propres) :

{profil}

---

{planning}

---

## Faits récents

{faits}"""
    _context_last_refresh = time.time()
    logger.info(f"Transcription context loaded: {len(_context_text)} chars")
    return _context_text


async def correct_transcription(raw_text: str) -> str:
    """Post-correct Whisper output using Claude Sonnet (proper nouns + homophones)."""
    if not raw_text or len(raw_text.strip()) < 5:
        return raw_text

    ctx = _get_context()
    if not ctx:
        return raw_text

    prompt = f"""{ctx}

---

## Transcription brute (message vocal Telegram, Whisper)

{raw_text}

## Tâche

Corrige UNIQUEMENT les noms propres mal transcrits et les homophones évidents.
Ne modifie PAS le sens, ne résume pas, ne reformule pas, n'ajoute rien.
Si la transcription est correcte, retourne-la telle quelle.
Retourne UNIQUEMENT le texte corrigé, sans explication ni commentaire."""

    try:
        env = dict(os.environ)
        env.pop("CLAUDECODE", None)

        proc = await asyncio.create_subprocess_exec(
            "claude", "-p", prompt,
            "--output-format", "json",
            "--model", "sonnet",
            "--tools", "",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=tempfile.gettempdir(),
            env=env,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)

        if proc.returncode != 0:
            logger.warning(f"Sonnet correction failed: {stderr.decode()[:200]}")
            return raw_text

        try:
            data = json.loads(stdout.decode())
            corrected = data.get("result", "").strip()
        except json.JSONDecodeError:
            corrected = stdout.decode().strip()

        # Sanity check: reject hallucinations
        if corrected and 0.3 < len(corrected) / len(raw_text) < 3.0:
            if corrected != raw_text:
                logger.info(f"Transcription corrected: '{raw_text[:80]}' → '{corrected[:80]}'")
            return corrected
        else:
            logger.warning(f"Correction rejected (length ratio): {len(corrected)}/{len(raw_text)}")
            return raw_text
    except asyncio.TimeoutError:
        logger.warning("Sonnet correction timed out (15s)")
        return raw_text
    except Exception as e:
        logger.warning(f"Sonnet correction failed: {e}")
        return raw_text

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
                data={"model": "voxtral-mini-2602", "language": "fr"},
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
    """Transcribe audio — pick engine based on duration, then post-correct."""
    wav_path = convert_to_wav(audio_path)
    try:
        duration = get_audio_duration(wav_path)
        if duration < DURATION_THRESHOLD:
            result = transcribe_whisper(wav_path)
        else:
            result = await transcribe_voxtral(audio_path)

        # Post-correct with Sonnet (proper nouns + homophones)
        corrected = await correct_transcription(result.text)
        if corrected != result.text:
            result = TranscriptionResult(
                text=corrected,
                engine=f"{result.engine}+sonnet",
                duration=result.duration,
                duration_formatted=result.duration_formatted,
            )
        return result
    finally:
        try:
            Path(wav_path).unlink(missing_ok=True)
        except Exception:
            pass
