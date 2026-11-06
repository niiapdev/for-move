import asyncio
import logging
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Awaitable, Callable, Optional

import ffmpeg

import config

logger = logging.getLogger("merger_local")

LOCAL_STAGE_DIR = Path(__file__).resolve().parent / "local_stage"

_SUBTITLE_STYLE = (
    "FontName=Arial,FontSize=20,PrimaryColour=&HFFFFFF,"
    "OutlineColour=&H000000,BackColour=&H80000000,Bold=0,"
    "Outline=1.5,Shadow=0.5,MarginV=30,Alignment=2"
)


class MergerError(RuntimeError):
    pass


async def merge_video_subtitles(
    video_path: Path,
    srt_path: Path,
    translate_to_ru: bool = False,
    progress_coro: Optional[Callable[[str], Awaitable[None]]] = None,
    out_name: Optional[str] = None,
    watermark_path: Optional[Path] = None,
) -> Path:
    if progress_coro is not None:
        await progress_coro("⏳ Склейка видео с субтитрами...")
    LOCAL_STAGE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOCAL_STAGE_DIR / (out_name or f"{video_path.stem}_subtitled.mp4")

    def _run_burn() -> None:
        safe_srt = srt_path.with_name(f"subs_{uuid.uuid4().hex}.srt")
        shutil.copyfile(srt_path, safe_srt)
        try:
            if watermark_path is None:
                vf = f"subtitles={safe_srt.name}:force_style='{_SUBTITLE_STYLE}'"
                cmd = [
                    "ffmpeg", "-y",
                    "-i", str(video_path),
                    "-vf", vf,
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "19",
                    "-c:a", "copy",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    str(out_path),
                ]
            else:
                info = ffmpeg.probe(str(video_path))
                streams = [s for s in info["streams"] if s.get("codec_type") == "video"]
                height = int(streams[0]["height"]) if streams else 720
                logo_h = max(40, round(height * 0.10))
                if logo_h % 2:
                    logo_h += 1
                fc = (
                    f"[0:v]subtitles={safe_srt.name}:force_style='{_SUBTITLE_STYLE}'[base];"
                    f"[1:v]scale=-2:{logo_h}[wm];[base][wm]overlay=16:16[out]"
                )
                cmd = [
                    "ffmpeg", "-y",
                    "-i", str(video_path),
                    "-i", str(watermark_path),
                    "-filter_complex", fc,
                    "-map", "[out]",
                    "-map", "0:a?",
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "19",
                    "-c:a", "copy",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    str(out_path),
                ]
            proc = subprocess.run(
                cmd,
                cwd=str(safe_srt.parent),
                capture_output=True,
                text=True,
                errors="replace",
            )
            if proc.returncode != 0:
                detail = (proc.stderr or "").strip().splitlines()
                raise MergerError(detail[-1][:300] if detail else "Ошибка ffmpeg")
        finally:
            safe_srt.unlink(missing_ok=True)

    def _run_mux() -> None:
        try:
            language = "rus" if translate_to_ru else "eng"
            video = ffmpeg.input(str(video_path))
            subs = ffmpeg.input(str(srt_path))
            stream = ffmpeg.output(
                video,
                subs,
                str(out_path),
                **{
                    "c:v": "copy",
                    "c:a": "copy",
                    "c:s": "mov_text",
                    "metadata:s:s:0": f"language={language}",
                },
            )
            ffmpeg.run(stream, overwrite_output=True, quiet=True, capture_stderr=True)
        except ffmpeg.Error as exc:
            detail = exc.stderr.decode("utf-8", "ignore").strip().splitlines()
            raise MergerError(detail[-1][:300] if detail else "Ошибка ffmpeg") from exc

    def _run() -> None:
        if config.BURN_SUBTITLES:
            _run_burn()
        else:
            _run_mux()

    await asyncio.to_thread(_run)
    if progress_coro is not None:
        await progress_coro("✅ Склейка завершена")
    return out_path