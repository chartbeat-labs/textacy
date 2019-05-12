"""
Language Detection
------------------
"""
# TODO: identify the "best" language detector available for OSS python
# in terms of prediction accuracy/precision/recall, speed, and installation
# current candidates:
# - cld2-cffi https://github.com/GregBowyer/cld2-cffi
# - pycld2 https://github.com/aboSamoor/pycld2
# - cld3 https://github.com/google/cld3 or https://github.com/Elizafox/cld3
# - langdetect https://github.com/Mimino666/langdetect
# - whatthelang https://github.com/indix/whatthelang
# - langid https://github.com/saffsd/langid.py
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

try:
    from cld2 import detect as cld2_detect
except ImportError:
    pass

from . import compat

LOGGER = logging.getLogger(__name__)


def detect_lang(text):
    """
    Predict the most likely language of a text and return its 2-letter code
    (see https://cloud.google.com/translate/v2/using_rest#language-params).
    Uses the `cld2-cffi <https://pypi.python.org/pypi/cld2-cffi>`_ package;
    to take advantage of optional params, call :func:`cld2.detect()` directly.

    Args:
        text (str)

    Returns:
        str
    """
    try:
        cld2_detect
    except NameError:
        raise ImportError(
            "`cld2-cffi` must be installed to use textacy's automatic language detection; "
            "you may do so via `pip install cld2-cffi` or `pip install textacy[lang]`."
        )
    if compat.PY2:
        is_reliable, _, best_guesses = cld2_detect(
            compat.to_bytes(text), bestEffort=True
        )
    else:
        is_reliable, _, best_guesses = cld2_detect(text, bestEffort=True)
    if is_reliable is False:
        LOGGER.warning(
            "Text language detected with low confidence; best guesses: %s",
            best_guesses,
        )
    return best_guesses[0][1]
