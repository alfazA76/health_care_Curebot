# translator.py
import os
import requests
from typing import Optional

try:
    from googletrans import Translator as GoogleTranslator  # unofficial Google Translate API
except Exception:  # pragma: no cover
    GoogleTranslator = None

try:
    from langdetect import detect as ld_detect  # fallback language detector
except Exception:  # pragma: no cover
    ld_detect = None

_ALLOWED_LANGS = {"en", "hi", "gu"}

_google_translator: Optional[GoogleTranslator] = None
if GoogleTranslator is not None:
    try:
        _google_translator = GoogleTranslator()
    except Exception:
        _google_translator = None


def _libretranslate(text: str, target: str, source: str = "auto") -> Optional[str]:
    """Fallback using LibreTranslate API. Public instance may be rate-limited.
    Configure custom instance via LIBRETRANSLATE_URL env (e.g., http://localhost:5000).
    """
    base = os.getenv("LIBRETRANSLATE_URL", "https://libretranslate.com")
    url = base.rstrip("/") + "/translate"
    try:
        resp = requests.post(
            url,
            timeout=12,
            json={"q": text, "source": source, "target": target, "format": "text"},
            headers={"Content-Type": "application/json"},
        )
        if resp.ok:
            data = resp.json()
            return data.get("translatedText")
    except Exception:
        pass
    return None


def detect_language(text: str) -> str:
    """Detect language code; prefer googletrans, fallback to langdetect, default to 'en'."""
    if not text:
        return "en"
    # Try googletrans
    if _google_translator is not None:
        try:
            return _google_translator.detect(text).lang or "en"
        except Exception:
            pass
    # Fallback to langdetect
    if ld_detect is not None:
        try:
            return ld_detect(text)
        except Exception:
            pass
    return "en"


essential_lang_map = {
    "en": "en",
    "hi": "hi",
    "gu": "gu",
}


def normalize_user_lang(code: str) -> str:
    """Map arbitrary language code to our supported set (en, hi, gu). Default to en."""
    code = (code or "en").split("-")[0].lower()
    return essential_lang_map.get(code, "en")


def translate_text(text: str, target_lang: str) -> str:
    """Translate text to target_lang using googletrans, fallback to LibreTranslate.
    target_lang must be two-letter code like 'en', 'hi', 'gu'.
    """
    if not text:
        return text
    # Try googletrans
    if _google_translator is not None:
        try:
            return _google_translator.translate(text, dest=target_lang).text
        except Exception:
            pass
    # Fallback to LibreTranslate
    lt = _libretranslate(text, target=target_lang)
    if lt is not None:
        return lt
    # Last resort: return original
    return text
