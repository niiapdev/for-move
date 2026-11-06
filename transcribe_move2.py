import asyncio
import logging
import re
import shutil
from pathlib import Path

from vkbottle import Bot, BotTypes

from config import OUTPUT_DIR, VK_TOKEN
from downloader import download_video
from pipeline import DownloadPipelineError
from queue_manager import Job, QueueManager
from translator import transcribe_video

import srt_translate

logger = logging.getLogger("vk_transcribe_move2")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


class TranscribeMove2Queue(QueueManager):
    async def _process_job(self, job: Job) -> str:
        progress = job.progress_coro
        if progress is not None:
            await progress("⬇️ Скачивание видео...")
        video_path, title_or_error = await download_video(job.url, progress)
        if video_path is None:
            raise DownloadPipelineError(title_or_error or "Не удалось скачать видео")
        srt_path = await transcribe_video(
            video_path,
            source_lang=None,
            progress_coro=progress,
            translate_to_en=True,
        )
        output_video = OUTPUT_DIR / video_path.name
        output_srt = OUTPUT_DIR / srt_path.name
        shutil.move(str(video_path), str(output_video))
        shutil.move(str(srt_path), str(output_srt))
        langs = await srt_translate.translate_srt(
            output_srt,
            progress_coro=progress,
        )
        done = sorted(lang for lang, path in langs.items() if path is not None)
        failed = sorted(lang for lang, path in langs.items() if path is None)
        if progress is not None:
            if failed:
                await progress(f"⚠️ Не удалось перевести: {', '.join(failed)}")
            await progress(f"✅ Создано: EN + {', '.join(done)}")
        return str(output_srt)


queue = TranscribeMove2Queue()

bot = Bot(VK_TOKEN)


async def _notify_done(job: Job) -> None:
    try:
        await bot.api.messages.send(peer_id=job.user_id, random_id=0, message=f"✅ Готово! Файл: {job.result}")
    except Exception as exc:
        logger.exception("Ошибка отправки уведомления о готовности: %s", exc)


async def _notify_progress(message: str) -> None:
    job = queue.running_job
    if job is None:
        return
    try:
        await bot.api.messages.send(peer_id=job.user_id, random_id=0, message=message)
    except Exception as exc:
        logger.exception("Ошибка отправки прогресса: %s", exc)


@bot.on.message()
async def message_handler(message: BotTypes.Message) -> None:
    text = (message.text or "").strip()
    user_id = message.peer_id

    if text == "/cancel":
        queue.cancel_user_jobs(user_id)
        await message.answer("❌ Ваши задачи в очереди отменены")
        return

    if not re.match(r"^https?://", text, flags=re.IGNORECASE):
        await message.answer("Пришлите ссылку на видео (http/https).")
        return

    queue.submit(
        url=text,
        user_id=user_id,
        translate_to_ru=False,
        merge=False,
        notify_coro=_notify_done,
        progress_coro=_notify_progress,
    )
    await message.answer("✅ Ссылка принята! Встало в очередь!")


if __name__ == "__main__":
    queue.set_default_notifiers(_notify_done, _notify_progress)
    logger.info("Бот запущен")
    asyncio.run(bot.run_polling())