import pytest

from textacy import preprocessing


@pytest.mark.parametrize(
    "text_in, fast, text_out",
    [
        (
            "El niño se asustó del pingüino -- qué miedo!",
            True,
            "El nino se asusto del pinguino -- que miedo!",
        ),
        (
            "El niño se asustó del pingüino -- qué miedo!",
            False,
            "El nino se asusto del pinguino -- que miedo!",
        ),
        (
            "Le garçon est très excité pour la forêt.",
            True,
            "Le garcon est tres excite pour la foret.",
        ),
        (
            "Le garçon est très excité pour la forêt.",
            False,
            "Le garcon est tres excite pour la foret.",
        ),
    ]
)
def test_remove_accents(text_in, fast, text_out):
    assert preprocessing.remove.accents(text_in, fast=fast) == text_out


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
    assert preprocessing.remove.brackets(text_in, only=only) == text_out


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
    assert preprocessing.remove.html_tags(text_in) == text_out


@pytest.mark.parametrize(
    "text_in, only, text_out",
    [
        (
            "I can't. No, I won't! It's a matter of \"principle\"; of -- what's the word? -- conscience.",
            None,
            "I can t  No  I won t  It s a matter of  principle   of    what s the word     conscience ",
        ),
        (
            "I can't. No, I won't! It's a matter of \"principle\"; of -- what's the word? -- conscience.",
            ".",
            "I can't  No, I won't! It's a matter of \"principle\"; of -- what's the word? -- conscience ",
        ),
        (
            "I can't. No, I won't! It's a matter of \"principle\"; of -- what's the word? -- conscience.",
            ["-", "'", "\""],
            "I can t. No, I won t! It s a matter of  principle ; of   what s the word?   conscience.",
        ),
    ]
)
def test_remove_punct(text_in, only, text_out):
    assert preprocessing.remove.punctuation(text_in, only=only) == text_out
