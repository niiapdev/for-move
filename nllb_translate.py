import threading
from pathlib import Path
from typing import Optional

import ctranslate2
import sentencepiece as spm

MODEL_DIR = Path(__file__).parent / "models" / "nllb-600m-int8"
SRC_LANG = "eng_Latn"
BEAM = 1
MAX_DECODING_LENGTH = 512

FLORES_CODES = {"ru": "rus_Cyrl", "kk": "kaz_Cyrl", "zh": "zho_Hans", "ja": "jpn_Jpan"}

_lock = threading.Lock()
_sp: Optional[spm.SentencePieceProcessor] = None
_translator = None
_loaded = False


def _ensure() -> None:
    global _sp, _translator, _loaded
    if _loaded:
        return
    with _lock:
        if _loaded:
            return
        sp = spm.SentencePieceProcessor()
        sp.load(str(MODEL_DIR / "sentencepiece.bpe.model"))
        translator = ctranslate2.Translator(str(MODEL_DIR), device="cpu", compute_type="int8")
        _sp, _translator, _loaded = sp, translator, True


def _strip_lang_leak(text: str, lang_token: str) -> str:
    if text.startswith(lang_token):
        rest = text[len(lang_token):].lstrip()
        return rest if rest else text
    return text


def translate_lines(texts: list, lang: str) -> list:
    if lang not in FLORES_CODES:
        raise ValueError(f"Язык {lang} не поддерживается NLLB")
    _ensure()
    target = FLORES_CODES[lang]
    source = []
    for pieces in _sp.encode_as_pieces(texts):
        source.append([SRC_LANG] + pieces + ["</s>"])
    prefixes = [[target]] * len(texts)
    with _lock:
        results = _translator.translate_batch(
            source,
            batch_type="tokens",
            max_batch_size=1024,
            beam_size=BEAM,
            max_decoding_length=MAX_DECODING_LENGTH,
            target_prefix=prefixes,
        )
    out = []
    for r in results:
        text = _sp.decode(r.hypotheses[0]).strip()
        text = _strip_lang_leak(text, target)
        out.append(text)
    return out