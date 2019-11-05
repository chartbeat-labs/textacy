import pytest

from textacy import preprocessing


def test_normalize_hyphenated_words():
    in_outs = [
        ("I see you shiver with antici- pation.", "I see you shiver with anticipation."),
        ("I see you shiver with antici-   \npation.", "I see you shiver with anticipation."),
        ("I see you shiver with antici- PATION.", "I see you shiver with anticiPATION."),
        ("I see you shiver with antici- 1pation.", "I see you shiver with antici- 1pation."),
        ("I see you shiver with antici pation.", "I see you shiver with antici pation."),
        ("I see you shiver with antici-pation.", "I see you shiver with antici-pation."),
        ("My phone number is 555- 1234.", "My phone number is 555- 1234."),
        ("I got an A- on the test.", "I got an A- on the test."),
    ]
    for in_, out_ in in_outs:
        assert preprocessing.normalize_hyphenated_words(in_) == out_


def test_normalize_quotation_marks():
    in_outs = [
        ("These are ´funny single quotes´.", "These are 'funny single quotes'."),
        ("These are ‘fancy single quotes’.", "These are 'fancy single quotes'."),
        ("These are “fancy double quotes”.", "These are \"fancy double quotes\"."),
    ]
    for in_, out_ in in_outs:
        assert preprocessing.normalize_quotation_marks(in_) == out_


def test_normalize_repeating_chars():
    text = "**Hello**, world!!! I wonder....... How are *you* doing?!?! lololol"
    kwargs_outputs = [
        (
            dict(chars=".", maxn=3),
            "**Hello**, world!!! I wonder... How are *you* doing?!?! lololol",
        ),
        (
            dict(chars="*", maxn=1),
            "*Hello*, world!!! I wonder....... How are *you* doing?!?! lololol",
        ),
        (
            dict(chars="?!", maxn=1),
            "**Hello**, world!!! I wonder....... How are *you* doing?! lololol",
        ),
        (
            dict(chars="ol", maxn=2),
            "**Hello**, world!!! I wonder....... How are *you* doing?!?! lolol",
        ),
    ]
    for kwargs, output in kwargs_outputs:
        assert preprocessing.normalize_repeating_chars(text, **kwargs) == output


def test_normalize_unicode():
    text = "Well… That's a long story."
    proc_text = "Well... That's a long story."
    assert preprocessing.normalize_unicode(text, form="NFKC") == proc_text


def test_normalize_whitespace():
    text = "Hello, world!  Hello...\t \tworld?\n\nHello:\r\n\n\nWorld. "
    proc_text = "Hello, world! Hello... world?\nHello:\nWorld."
    assert preprocessing.normalize_whitespace(text) == proc_text
