import asyncio
import logging
from pathlib import Path
from typing import Awaitable, Callable, Optional

from faster_whisper import WhisperModel

import config

logger = logging.getLogger("translator")

_model: Optional[WhisperModel] = None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )
    return _model


def _format_ts(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, rem = divmod(rem, 60)
    secs, millis = divmod(rem, 1)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d},{int(millis * 1000):03d}"


def _format_srt(blocks) -> str:
    lines = []
    for idx, (start, end, text) in enumerate(blocks, start=1):
        lines.append(str(idx))
        lines.append(f"{_format_ts(start)} --> {_format_ts(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


async def transcribe_video(
    video_path: Path,
    source_lang: str = "en",
    progress_coro: Optional[Callable[[str], Awaitable[None]]] = None,
    translate_to_ru: bool = False,
    translate_to_en: bool = False,
) -> Path:
    if progress_coro is not None:
        await progress_coro("⏳ Распознавание речи...")
    if translate_to_ru:
        language = "ru"
        task = "transcribe"
    elif translate_to_en:
        language = None
        task = "translate"
    else:
        language = source_lang or None
        task = "transcribe"
    srt_path = video_path.with_suffix(".srt")
    shift = float(config.SUBTITLE_SHIFT_SECONDS or 0.0)

    def _run() -> Path:
        model = _get_model()
        segments, _info = model.transcribe(
            str(video_path),
            language=language,
            task=task,
            word_timestamps=bool(config.WHISPER_WORD_TIMESTAMPS),
            vad_filter=bool(config.WHISPER_VAD),
        )
        blocks = []
        for segment in segments:
            text = (segment.text or "").strip()
            if not text:
                continue
            blocks.append((segment.start + shift, segment.end + shift, text))
        srt_path.write_text(_format_srt(blocks), encoding="utf-8")
        return srt_path

    return await asyncio.to_thread(_run)