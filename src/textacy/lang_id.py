import logging

import gcld3


LOGGER = logging.getLogger(__name__)

lang_identifier = gcld3.NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=1000)


def identify_lang(text: str) -> str:
    """
    Identify the most probable language identified in ``text``.

    Args:
        text

    Returns:
        BCP-47-style language code of the most probable language, e.g. ``"en"``.

    Warning:
        Language identification results are known to be unreliable
        for extremely short texts (fewer than ~20 characters).
    """
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
