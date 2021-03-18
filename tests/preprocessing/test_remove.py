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


@pytest.mark.parametrize(
    "text_in, only, text_out",
    [
        ("Hello, {name}!", None, "Hello, !"),
        ("Hello, world (DeWilde et al., 2021, p. 42)!", None, "Hello, world !"),
        ("Hello, world (1)!", None, "Hello, world !"),
        ("Hello, world [1]!", None, "Hello, world !"),
        (
            "Hello, world (and whomever it may concern [not that it's any of my business])!",
            None,
            "Hello, world !",
        ),
        (
            "Hello, world (and whomever it may concern (not that it's any of my business))!",
            None,
            "Hello, world (and whomever it may concern )!",
        ),
        (
            "Hello, world (and whomever it may concern [not that it's any of my business])!",
            "square",
            "Hello, world (and whomever it may concern )!",
        ),
        ("Hello, world [1]!", "round", "Hello, world [1]!"),
        ("Hello, world [1]!", ("curly", "round"), "Hello, world [1]!"),
    ]
)
def test_remove_brackets(text_in, only, text_out):
    assert preprocessing.remove.remove_brackets(text_in, only=only) == text_out


@pytest.mark.parametrize(
    "text_in, text_out",
    [
        ("Hello, <i>world!</i>", "Hello, world!"),
        ("<title>Hello, world!</title>", "Hello, world!"),
        ('<title class="foo">Hello, world!</title>', "Hello, world!"),
        (
            "<html><head><title>Hello, <i>world!</i></title></head></html>",
            "Hello, world!",
        ),
        (
            "<html>\n"
            "  <head>\n"
            '    <title class="foo">Hello, <i>world!</i></title>\n'
            "  </head>\n"
            "  <!--this is a comment-->\n"
            "  <body>\n"
            "    <p>How's it going?</p>\n"
            "  </body>\n"
            "</html>",
            "Hello, world!\n  \n  \n  \n    How's it going?",
        ),
    ]
)
def test_remove_html_tags(text_in, text_out):
    assert preprocessing.remove.remove_html_tags(text_in) == text_out
