# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

import pytest

from textacy import preprocessing


def test_remove_punct():
    text = "I can't. No, I won't! It's a matter of \"principle\"; of -- what's the word? -- conscience."
    proc_text = "I can t  No  I won t  It s a matter of  principle   of    what s the word     conscience "
    assert preprocessing.remove_punctuation(text) == proc_text


def test_remove_punct_marks():
    text = "I can't. No, I won't! It's a matter of \"principle\"; of -- what's the word? -- conscience."
    proc_text = "I can t. No, I won t! It s a matter of  principle ; of   what s the word?   conscience."
    assert preprocessing.remove_punctuation(text, marks="-'\"") == proc_text


def test_remove_accents():
    text = "El niño se asustó -- qué miedo!"
    proc_text = "El nino se asusto -- que miedo!"
    assert preprocessing.remove_accents(text, method="unicode") == proc_text
    assert preprocessing.remove_accents(text, method="ascii") == proc_text
    with pytest.raises(ValueError):
        _ = preprocessing.remove_accents(text, method="foo")
