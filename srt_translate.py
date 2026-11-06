import asyncio
import json
import logging
import random
import urllib.error
import urllib.request
from pathlib import Path
from typing import Awaitable, Callable, Optional

import nllb_translate

logger = logging.getLogger("srt_translate")

PROVIDER = "nllb"
YANDEX_API_KEY = ""
AZURE_SUBSCRIPTION_KEY = ""
AZURE_REGION = ""

TARGET_LANGS = ["ru", "kk", "zh", "ja"]
BATCH_SIZE = 40
MIN_DELAY = 0.4
MAX_DELAY = 3.0
RETRIES = 3
TIMEOUT = 60

YANDEX_URL = "https://translate.api.cloud.yandex.net/translate/v2/translate"
AZURE_URL = "https://api.cognitive.microsofttranslator.com/translate?api-version=3.0&to={lang}"

YANDEX_CODES = {"ru": "ru", "kk": "kk", "zh": "zh", "ja": "ja"}
AZURE_CODES = {"ru": "ru", "kk": "kk", "zh": "zh-Hans", "ja": "ja"}


class SrtTranslationError(RuntimeError):
    pass


def lang_suffix(lang: str) -> str:
    return lang.split("-")[0].upper()


def parse_srt(path: Path):
    data = path.read_text(encoding="utf-8")
    blocks = [b.strip() for b in data.split("\n\n") if b.strip()]
    cues = []
    for block in blocks:
        lines = block.splitlines()
        if not lines or not lines[0].strip().isdigit():
            continue
        if len(lines) < 3:
            continue
        number = int(lines[0].strip())
        timing = lines[1].strip()
        text = "\n".join(l.strip() for l in lines[2:]).strip()
        if text:
            cues.append((number, timing, text))
    return cues


def write_srt(path: Path, cues, out_texts) -> None:
    lines = []
    for (number, timing, original), out in zip(cues, out_texts):
        lines.append(str(number))
        lines.append(timing)
        lines.append(out if out else original)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _provider_codes() -> dict:
    if PROVIDER == "azure":
        return AZURE_CODES
    if PROVIDER == "yandex":
        return YANDEX_CODES
    return nllb_translate.FLORES_CODES


def _check_keys() -> None:
    if PROVIDER == "azure":
        if not AZURE_SUBSCRIPTION_KEY:
            raise SrtTranslationError("Не задан AZURE_SUBSCRIPTION_KEY в srt_translate.py")
    elif PROVIDER == "yandex" and not YANDEX_API_KEY:
        raise SrtTranslationError("Не задан YANDEX_API_KEY в srt_translate.py")


def _translate_batch(texts: list, code: str) -> list:
    if PROVIDER == "azure":
        payload = json.dumps([{"text": t} for t in texts]).encode()
        url = AZURE_URL.format(lang=code)
        headers = {
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": AZURE_SUBSCRIPTION_KEY,
        }
        if AZURE_REGION:
            headers["Ocp-Apim-Subscription-Region"] = AZURE_REGION
        body = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    else:
        payload = json.dumps({
            "sourceLanguageCode": "en",
            "targetLanguageCode": code,
            "texts": texts,
        }).encode()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {YANDEX_API_KEY}",
        }
        body = urllib.request.Request(YANDEX_URL, data=payload, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(body, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "ignore")[:200]
        except Exception:
            pass
        raise SrtTranslationError(f"HTTP {exc.code}: {detail}") from exc

    if PROVIDER == "azure":
        if not isinstance(data, list) or len(data) != len(texts):
            raise SrtTranslationError("Azure: неожиданный формат ответа")
        return [item["translations"][0]["text"] for item in data]
    translations = data.get("translations")
    if not isinstance(translations, list) or len(translations) != len(texts):
        raise SrtTranslationError("Yandex: неожиданный формат ответа")
    return [item["text"] for item in translations]


async def _request_batch_with_retries(texts: list, code: str) -> list:
    for attempt in range(RETRIES):
        delay = MIN_DELAY + random.uniform(0.0, 0.5) + attempt * 0.8
        await asyncio.sleep(delay)
        try:
            return await asyncio.to_thread(_translate_batch, texts, code)
        except SrtTranslationError as exc:
            logger.warning("Ошибка перевода (%s, попытка %d): %s", code, attempt + 1, exc)
            if attempt == RETRIES - 1:
                raise
    raise SrtTranslationError(f"Не удалось перевести на {code}")


async def translate_texts(texts, code: str, target_lang: str, progress_coro=None) -> list:
    results = [None] * len(texts)
    cache = {}
    i = 0
    total = len(texts)
    while i < total:
        batch_end = min(i + BATCH_SIZE, total)
        batch = list(range(i, batch_end))
        pending = [j for j in batch if texts[j] not in cache]
        pair = None
        if pending:
            pair = {}
            if PROVIDER == "nllb":
                try:
                    translated = await asyncio.to_thread(
                        nllb_translate.translate_lines, [texts[j] for j in pending], target_lang
                    )
                except Exception as exc:
                    logger.exception("Ошибка NLLB перевода на %s", target_lang)
                    translated = None
                if translated is not None:
                    for k, t in zip(pending, translated):
                        pair[texts[k]] = (t or "").strip()
                else:
                    pair = None
            else:
                try:
                    translated = await _request_batch_with_retries([texts[j] for j in pending], code)
                    for k, t in zip(pending, translated):
                        pair[texts[k]] = (t or "").strip()
                except SrtTranslationError:
                    pair = None
        for j in batch:
            if pair is not None:
                cache[texts[j]] = pair.get(texts[j]) or texts[j]
                results[j] = cache[texts[j]]
            else:
                results[j] = cache.get(texts[j]) or texts[j]
        i = batch_end
        if progress_coro is not None:
            await progress_coro(f"⏳ Перевод субтитров на {target_lang}... {int(i * 100 / total)}%")
    return results


async def translate_srt(
    srt_path: Path,
    target_langs: Optional[list] = None,
    progress_coro: Optional[Callable[[str], Awaitable[None]]] = None,
) -> dict:
    target_langs = target_langs or TARGET_LANGS
    if PROVIDER != "nllb":
        _check_keys()
    codes = _provider_codes()
    cues = parse_srt(srt_path)
    texts = [c[2] for c in cues]
    result = {}
    for lang in target_langs:
        try:
            if progress_coro is not None:
                await progress_coro(f"⏳ Перевод субтитров на {lang}...")
            code = codes[lang]
            out_texts = await translate_texts(texts, code, lang, progress_coro)
            out_path = srt_path.with_name(f"{srt_path.stem}.{lang_suffix(lang)}.srt")
            write_srt(out_path, cues, out_texts)
            if progress_coro is not None:
                await progress_coro(f"✅ Субтитры на {lang}: {out_path.name}")
            result[lang] = out_path
        except Exception as exc:
            logger.exception("Не удалось перевести на %s", lang)
            result[lang] = None
    return result