# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

from textacy import lang_utils

LANG_SENTS = [
    ("en", "This sentence is in English."),
    ("es", "Esta oración es en Español."),
    ("fr", "Cette phrase est en français."),
    ("un", "1"),
    ("un", " "),
    ("un", ""),
]


def test_detect_language():
    for lang, sent in LANG_SENTS:
        assert lang_utils.detect_lang(sent) == lang
