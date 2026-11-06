import asyncio
import json
import logging
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import ffmpeg

import config
import merger_local
import translator

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "local_videos"
STAGE_DIR = BASE_DIR / "local_stage"
OUT_DIR = BASE_DIR / "local_out"
STATE_PATH = STAGE_DIR / "segments.json"
LOGO_NAMES = ("logo.png", "logo.jpg", "logo.jpeg", "logo.webp")
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".wmv"}
REMERGE = "--remerge" in sys.argv

logger = logging.getLogger("local_bot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


async def _progress(message: str) -> None:
    print(message, flush=True)


def _find_logo() -> Optional[Path]:
    for name in LOGO_NAMES:
        path = INPUT_DIR / name
        if path.is_file():
            return path
    return None


def _logo_fingerprint(logo: Optional[Path]) -> str:
    if logo is None:
        return ""
    stat = logo.stat()
    return f"{logo.name}|{stat.st_size}|{int(stat.st_mtime)}"


def _load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_state(state: dict) -> None:
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _numbered_videos() -> list:
    result = []
    for path in INPUT_DIR.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        match = re.match(r"^\s*(\d+)", path.stem)
        if match:
            result.append((int(match.group(1)), path))
    return sorted(result, key=lambda item: (item[0], item[1].name))


def _video_size(path: Path) -> tuple:
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    values = proc.stdout.split()
    try:
        return int(values[0]), int(values[1])
    except (IndexError, ValueError):
        return 1920, 1080


def _build_concat(parts, output: Path) -> None:
    sizes = [_video_size(p) for p in parts]
    target_w = max(size[0] for size in sizes)
    target_h = max(size[1] for size in sizes)
    uniform = all(size == (target_w, target_h) for size in sizes)
    streams = []
    for path, size in zip(parts, sizes):
        src = ffmpeg.input(str(path))
        if uniform:
            video = src.video
        else:
            video = (
                src.video
                .filter("scale", target_w, target_h, force_original_aspect_ratio="decrease")
                .filter("pad", target_w, target_h, "(ow-iw)/2", "(oh-ih)/2")
                .filter("setsar", "1")
            )
        audio = src.audio.filter(
            "aformat", sample_fmts="fltp", sample_rates=48000, channel_layouts="stereo"
        )
        streams.append(video)
        streams.append(audio)
    joined = ffmpeg.concat(*streams, v=1, a=1)
    out = ffmpeg.output(
        joined,
        str(output),
        vcodec="libx264",
        preset="veryfast",
        crf="18",
        pix_fmt="yuv420p",
        acodec="aac",
        movflags="+faststart",
        **{"b:a": "192k"},
    )
    try:
        ffmpeg.run(out, overwrite_output=True, quiet=True, capture_stderr=True)
    except ffmpeg.Error as exc:
        detail = exc.stderr.decode("utf-8", "ignore").strip().splitlines()
        raise RuntimeError(detail[-1][:300] if detail else "Ошибка ffmpeg") from exc


def _stream_duration(path: Path, stream_type: str) -> float:
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", f"{stream_type}:0",
            "-show_entries", "stream=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return -1.0


def _validate_merged(parts, final: Path) -> tuple:
    sum_v = 0.0
    sum_a = 0.0
    for part in parts:
        sum_v += _stream_duration(part, "v")
        sum_a += _stream_duration(part, "a")
    final_v = _stream_duration(final, "v")
    final_a = _stream_duration(final, "a")
    reasons = []
    if sum_v > 0 and abs(final_v - sum_v) > 1.0:
        reasons.append(f"видео {final_v:.1f}с != сумма сегментов {sum_v:.1f}с")
    if final_v > 0 and final_a > 0 and abs(final_v - final_a) > 1.0:
        reasons.append(f"рассинхрон v={final_v:.1f}с / a={final_a:.1f}с")
    return (not reasons), "; ".join(reasons)


async def main() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logo = _find_logo()
    if logo is not None and not config.BURN_SUBTITLES:
        print("Внимание: водяной знак накладывается только при склейке с прожигом (BURN_SUBTITLES=True)")
    if logo is not None:
        print(f"Логотип: {logo.name} — будет накладываться на каждое видео")
    else:
        print("logo.png не найден — водяной знак накладываться не будет")

    state = _load_state()
    logo_fp = _logo_fingerprint(logo)
    items = _numbered_videos()

    seen = {}
    for number, path in items:
        seen.setdefault(number, []).append(path)
    for number, paths in seen.items():
        if len(paths) > 1:
            print(
                f"Внимание: номер {number} присвоен нескольким файлам — порядок по имени: "
                + ", ".join(p.name for p in paths)
            )

    if REMERGE:
        print("Режим --remerge: сборка финала из существующих сегментов local_stage/")
        done_numbers = []
        for seg in sorted(STAGE_DIR.glob("*_subtitled.mp4"), key=lambda p: p.name):
            match = re.match(r"^(\d+)_subtitled\.mp4$", seg.name)
            if match:
                done_numbers.append(int(match.group(1)))
    else:
        done_numbers = []
        for number, path in items:
            out_file = STAGE_DIR / f"{number}_subtitled.mp4"
            source_fp = {
                "src": path.name,
                "size": path.stat().st_size,
                "mtime": round(path.stat().st_mtime, 2),
            }
            if out_file.is_file() and state.get(str(number)) == {**source_fp, "logo": logo_fp}:
                print(f"{number}: {path.name} — пропуск (уже готово)")
                done_numbers.append(number)
                continue
            print(f"{number}/{len(items)}: {path.name} — распознавание речи...")
            try:
                srt_path = await translator.transcribe_video(path, progress_coro=_progress, translate_to_en=True)
                if srt_path.stat().st_size == 0:
                    print(f"{number}: речи не обнаружено — копирую видео без субтитров")
                    shutil.copyfile(path, out_file)
                else:
                    print(f"{number}: перевод в английские субтитры и склейка...")
                    await merger_local.merge_video_subtitles(
                        path,
                        srt_path,
                        out_name=out_file.name,
                        watermark_path=logo,
                        progress_coro=_progress,
                    )
            except Exception as exc:
                logger.exception("Ошибка обработки %s: %s", path.name, exc)
                continue
            finally:
                (path.with_suffix(".srt")).unlink(missing_ok=True)
            state[str(number)] = {**source_fp, "logo": logo_fp}
            _save_state(state)
            done_numbers.append(number)
            print(f"{number}: готово -> {out_file.name}")
        if not items:
            print(
                f"В {INPUT_DIR.name}/ нет пронумерованных видео. Положи 1.mp4, 2.mp4, ... "
                "(или запусти с --remerge для пересборки финала из сегментов)."
            )

    parts = [STAGE_DIR / f"{n}_subtitled.mp4" for n in sorted(done_numbers)]
    parts = [p for p in parts if p.is_file()]
    if not parts:
        print("Нет сегментов для склейки")
        return

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    final = OUT_DIR / f"merged_{stamp}.mp4"
    print(f"Склейка {len(parts)} видео в порядке нумерации (перекодирование, родные fps) -> {final.name}...")
    await asyncio.to_thread(_build_concat, parts, final)
    ok, reasons = _validate_merged(parts, final)
    if not ok:
        logger.error("Итоговый файл рассинхронизирован: %s — пересобери и проверь вручную", reasons)
    print(f"Итоговый файл: {final}")

    for number in sorted(done_numbers):
        src = STAGE_DIR / f"{number}_subtitled.mp4"
        if src.is_file():
            try:
                shutil.copyfile(src, OUT_DIR / src.name)
            except OSError as exc:
                logger.warning("Не удалось скопировать %s: %s", src.name, exc)
    used = {f"{n}_subtitled.mp4" for n in done_numbers}
    for stale in OUT_DIR.glob("*_subtitled.mp4"):
        if stale.name not in used:
            try:
                stale.unlink()
            except OSError as exc:
                logger.warning("Не удалось удалить старую копию %s: %s", stale.name, exc)
    print("Копии готовых сегментов скопированы в local_out/")

    if not REMERGE:
        for number in sorted(done_numbers):
            for path in seen[number]:
                try:
                    path.unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning("Не удалось удалить %s: %s", path.name, exc)
        print("Пронумерованные исходники удалены")


if __name__ == "__main__":
    asyncio.run(main())