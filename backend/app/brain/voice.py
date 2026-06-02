import base64
import json
import subprocess
import uuid
from pathlib import Path
from typing import Optional

import google.generativeai as genai

from app.config import (
    GEMINI_MODEL,
    MAX_MEDIA_MB,
    MAX_VOICE_SECONDS,
    VOICE_TRANSCRIPTION_FALLBACK,
    VOICE_TRANSCRIPTION_PROVIDER,
    WHISPER_MODEL_SIZE,
)
from app.brain.safety import get_workspace_dir


VOICE_DIR = Path(get_workspace_dir()) / "backend" / "local_data" / "voice_notes"


def _media_size_error(raw: bytes) -> Optional[str]:
    max_bytes = MAX_MEDIA_MB * 1024 * 1024
    if len(raw) > max_bytes:
        return f"Voice note is too large ({len(raw) / 1024 / 1024:.1f} MB). Limit is {MAX_MEDIA_MB} MB."
    return None


def _save_voice_media(media_base64: str, filename: Optional[str], mime_type: Optional[str]) -> Path:
    raw = base64.b64decode(media_base64)
    size_error = _media_size_error(raw)
    if size_error:
        raise ValueError(size_error)

    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    ext = Path(filename or "").suffix
    if not ext:
        mime = (mime_type or "").lower()
        if "ogg" in mime or "opus" in mime:
            ext = ".ogg"
        elif "mpeg" in mime or "mp3" in mime:
            ext = ".mp3"
        elif "wav" in mime:
            ext = ".wav"
        else:
            ext = ".audio"
    target = VOICE_DIR / f"{uuid.uuid4().hex[:10]}{ext}"
    target.write_bytes(raw)
    return target


def _probe_duration_seconds(path: Path) -> Optional[float]:
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode != 0:
            return None
        data = json.loads(proc.stdout or "{}")
        duration = data.get("format", {}).get("duration")
        return float(duration) if duration is not None else None
    except Exception:
        return None


def _convert_to_wav(path: Path) -> Path:
    duration = _probe_duration_seconds(path)
    if duration and duration > MAX_VOICE_SECONDS:
        raise ValueError(f"Voice note is {duration:.0f}s long. Limit is {MAX_VOICE_SECONDS}s.")

    wav_path = path.with_suffix(".wav")
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(path),
            "-t",
            str(MAX_VOICE_SECONDS),
            "-ar",
            "16000",
            "-ac",
            "1",
            str(wav_path),
        ],
        capture_output=True,
        text=True,
        timeout=max(45, MAX_VOICE_SECONDS + 15),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg could not convert the WhatsApp voice note. Install ffmpeg and make sure it is on PATH."
        )
    return wav_path


def _transcribe_with_local_whisper(audio_path: Path) -> str:
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise RuntimeError(
            "Local voice transcription needs `faster-whisper`. Install backend requirements first."
        ) from e

    wav_path = _convert_to_wav(audio_path)
    model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(wav_path), vad_filter=True)
    text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
    language = getattr(info, "language", "unknown")
    if not text:
        return f"No clear speech was detected. Detected language: {language}."
    return f"{text}\n\nDetected language: {language}"


def _transcribe_with_gemini(audio_path: Path) -> str:
    uploaded = genai.upload_file(path=str(audio_path))
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content([
        "Transcribe this WhatsApp voice note. Return only the transcript, then one short line with detected language.",
        uploaded
    ])
    return (response.text or "").strip()


def transcribe_voice_note(media_base64: str, mime_type: Optional[str] = None, filename: Optional[str] = None) -> str:
    if not media_base64:
        return "I received a voice note, but WhatsApp did not send the audio data."

    try:
        audio_path = _save_voice_media(media_base64, filename, mime_type)
    except Exception as e:
        return f"I received your voice note, but I could not save the audio for transcription.\n\nProblem: {e}"

    provider = VOICE_TRANSCRIPTION_PROVIDER

    if provider == "gemini":
        try:
            return _transcribe_with_gemini(audio_path)
        except Exception as e:
            return f"I received your voice note, but Gemini transcription failed.\n\nProblem: {e}"

    try:
        return _transcribe_with_local_whisper(audio_path)
    except Exception as local_error:
        if VOICE_TRANSCRIPTION_FALLBACK == "gemini":
            try:
                gemini_text = _transcribe_with_gemini(audio_path)
                return f"{gemini_text}\n\nNote: Local Whisper failed, so Gemini fallback was used."
            except Exception as gemini_error:
                return (
                    "I received your voice note, but transcription failed.\n\n"
                    f"Local Whisper error: {local_error}\n"
                    f"Gemini fallback error: {gemini_error}"
                )
        return (
            "I received your voice note, but free local transcription is not ready yet.\n\n"
            f"Problem: {local_error}\n\n"
            "Install `ffmpeg` and backend requirements, then try again. I did not use Gemini automatically."
        )
