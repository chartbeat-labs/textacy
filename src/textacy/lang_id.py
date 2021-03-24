"""
Language Identification
-----------------------

:mod:`textacy.lang_id`: Simple interface for identifying the most probable language
of a given text.
"""
import logging

import gcld3


LOGGER = logging.getLogger(__name__)

lang_identifier = gcld3.NNetLanguageIdentifier(min_num_bytes=3, max_num_bytes=1000)


def identify_lang(text: str) -> str:
    """
    Identify the most probable language identified in ``text``,
    or "un" (for "unknown") if one can't be identified.

    Args:
        text

    Returns:
        BCP-47-style language code of the most probable language, e.g. ``"en"``.

    Warning:
        Language identification results are known to be unreliable
        for extremely short texts (fewer than ~20 characters).
    """
    # gcld3 likes to guess "ja" (japanese) for short, non-alpha texts
    # so here we shield users from a GIGO situation
    if not _is_valid_text(text[:1000]):
        return "un"

    result = lang_identifier.FindLanguage(text)
    if result.is_reliable is False:
        LOGGER.warning("unable to reliably identify language of text = %s", text[:1000])
    # textacy's home-brewed lang identifier used "un" to indicate "undefined"/"unknown"
    # language identification results, while google's cld3 uses "und"
    # let's shield users from that inconsistency
    if result.language == "und":
        return "un"
    else:
        return result.language


def _is_valid_text(text: str) -> bool:
    return any(char.isalpha() for char in text)
