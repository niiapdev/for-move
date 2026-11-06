import asyncio
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Awaitable, Callable, Optional

import yt_dlp
from yt_dlp.utils import DownloadError

from config import WORK_DIR

logger = logging.getLogger("downloader")


def _readable_error(exc: BaseException) -> str:
    cause = getattr(exc, "excinfo", None) or exc
    raw = getattr(cause, "msg", None) or getattr(exc, "msg", None) or str(exc)
    raw = re.sub(r"^ERROR:\s*", "", str(raw).strip())
    raw = re.split(r"\s*\(caused by", raw, maxsplit=1)[0].strip()
    low = raw.lower()
    if "unsupported url" in low:
        return "Ссылка не поддерживается"
    if "private" in low:
        return "Видео приватное"
    if "sign in" in low or "login" in low:
        return "Для просмотра требуется авторизация"
    if "too many requests" in low or "timed out" in low:
        return "Сеть перегружена, попробуйте позже"
    return raw[:500] or "Неизвестная ошибка скачивания"


def _final_path(prefix: str) -> Optional[Path]:
    skipped = {".part", ".ytdl", ".temp"}
    candidates = [
        p
        for p in WORK_DIR.glob(f"{prefix}-*")
        if p.is_file() and p.suffix.lower() not in skipped
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p.stat().st_mtime, p.stat().st_size))


async def download_video(
    url: str,
    progress_coro: Optional[Callable[[str], Awaitable[None]]] = None,
) -> tuple[Optional[Path], Optional[str]]:
    loop = asyncio.get_running_loop()
    prefix = uuid.uuid4().hex[:12]
    last_update = {"t": 0.0, "finished": False}

    def _hook(data: dict) -> None:
        if progress_coro is None:
            return
        try:
            status = data.get("status")
            if status == "downloading":
                return
            elif status == "finished" and not last_update["finished"]:
                last_update["finished"] = True
                asyncio.run_coroutine_threadsafe(
                    progress_coro("⏳ Скачивание завершено"), loop
                )
        except Exception:
            logger.exception("Ошибка в progress hook")

    def _run() -> tuple[Optional[Path], Optional[str]]:
        try:
            opts = {
                "format": "best",
                "retries": 3,
                "merge_output_format": "mp4",
                "outtmpl": str(WORK_DIR / f"{prefix}-%(id)s.%(ext)s"),
                "noplaylist": True,
                "quiet": True,
                "no_warnings": True,
                "progress_hooks": [_hook] if progress_coro is not None else [],
            }
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
            if info is None:
                return None, "Не удалось получить информацию о видео"
            path = _final_path(prefix)
            if path is None:
                return None, "Файл не был сохранён на диск"
            title = str(info.get("title") or "video")
            return path, title
        except DownloadError as exc:
            return None, _readable_error(exc)
        except Exception as exc:
            logger.exception("Ошибка скачивания: %s", url)
            return None, _readable_error(exc)

    video_path, title_or_error = await asyncio.to_thread(_run)
    if video_path is None and progress_coro is not None:
        try:
            await progress_coro(f"❌ Ошибка скачивания: {title_or_error}")
        except Exception:
            logger.exception("Не удалось отправить сообщение об ошибке скачивания")
    return video_path, title_or_error