"""
Transliteration module supporting optional cross-script transliteration with graceful fallback.
"""

import logging
import unicodedata

logger = logging.getLogger(__name__)

# Check optional transliteration packages
HAS_UNIDECODE = False
try:
    from unidecode import unidecode
    HAS_UNIDECODE = True
except ImportError:
    pass

HAS_INDIC = False
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
    HAS_INDIC = True
except ImportError:
    pass


def transliterate_text(text: str, source_script: str = None, enabled: bool = True) -> str:
    """
    Transliterate non-Latin text to Latin representations.
    Fails gracefully if optional dependencies are not installed or transliteration is disabled.
    """
    if not text or not enabled:
        return text

    # If already pure ASCII, return as is
    if text.isascii():
        return text

    # 1. Try Indic Transliteration if script is Indic and library available
    if HAS_INDIC and source_script in ["Devanagari", "Telugu", "Tamil", "Kannada", "Bengali", "Gujarati", "Malayalam"]:
        script_scheme_map = {
            "Devanagari": sanscript.DEVANAGARI,
            "Telugu": sanscript.TELUGU,
            "Tamil": sanscript.TAMIL,
            "Kannada": sanscript.KANNADA,
            "Bengali": sanscript.BENGALI,
            "Gujarati": sanscript.GUJARATI,
            "Malayalam": sanscript.MALAYALAM,
        }
        scheme = script_scheme_map.get(source_script)
        if scheme:
            try:
                res = transliterate(text, scheme, sanscript.ITRANS)
                if res and res != text:
                    return res.lower()
            except Exception as e:
                logger.debug(f"Indic transliteration failed for '{text}': {e}")

    # 2. Try unidecode if available
    if HAS_UNIDECODE:
        try:
            res = unidecode(text)
            if res:
                return res.lower()
        except Exception as e:
            logger.debug(f"Unidecode transliteration failed for '{text}': {e}")

    # 3. Standard Unicode NFKD ASCII decomposition fallback
    try:
        nfkd = unicodedata.normalize("NFKD", text)
        ascii_text = "".join([c for c in nfkd if not unicodedata.combining(c)])
        if ascii_text.isascii() and ascii_text.strip():
            return ascii_text.lower()
    except Exception:
        pass

    # Return original text if transliteration unsupported
    return text
