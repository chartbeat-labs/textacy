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
    in_outs = [
        ("El niño se asustó del pingüino -- qué miedo!", "El nino se asusto del pinguino -- que miedo!"),
        ("Le garçon est très excité pour la forêt.", "Le garcon est tres excite pour la foret."),
    ]
    for in_, out_ in in_outs:
        assert preprocessing.remove_accents(in_, fast=False) == out_
        assert preprocessing.remove_accents(in_, fast=True) == out_
