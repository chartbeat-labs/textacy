import pytest

from textacy import preprocessing


@pytest.mark.parametrize(
    "text_in, text_out",
    [
        ("• foo\n• bar", "- foo\n- bar"),
        ("• foo\n    • bar", "- foo\n    - bar"),
        (
            "\n‣ item1\n⁃ item2\n⁌ item3\n⁍ item4\n∙ item5\n▪ item6\n● item7\n◦ item8",
            "\n- item1\n- item2\n- item3\n- item4\n- item5\n- item6\n- item7\n- item8",
        ),
        (
            "\n⦾ item1\n⦿ item2\n・ item3",
            "\n- item1\n- item2\n- item3",
        ),
    ]
)
def test_normalize_bullet_points(text_in, text_out):
    assert preprocessing.normalize.bullet_points(text_in) == text_out


@pytest.mark.parametrize(
    "text_in, text_out",
    [
        ("I see you shiver with antici- pation.", "I see you shiver with anticipation."),
        ("I see you shiver with antici-   \npation.", "I see you shiver with anticipation."),
        ("I see you shiver with antici- PATION.", "I see you shiver with anticiPATION."),
        ("I see you shiver with antici- 1pation.", "I see you shiver with antici- 1pation."),
        ("I see you shiver with antici pation.", "I see you shiver with antici pation."),
        ("I see you shiver with antici-pation.", "I see you shiver with antici-pation."),
        ("My phone number is 555- 1234.", "My phone number is 555- 1234."),
        ("I got an A- on the test.", "I got an A- on the test."),
    ]
)
def test_normalize_hyphenated_words(text_in, text_out):
    assert preprocessing.normalize.hyphenated_words(text_in) == text_out


@pytest.mark.parametrize(
    "text_in, text_out",
    [
        ("These are ´funny single quotes´.", "These are 'funny single quotes'."),
        ("These are ‘fancy single quotes’.", "These are 'fancy single quotes'."),
        ("These are “fancy double quotes”.", "These are \"fancy double quotes\"."),
    ]
)
def test_normalize_quotation_marks(text_in, text_out):
    assert preprocessing.normalize.quotation_marks(text_in) == text_out


@pytest.mark.parametrize(
    "kwargs, text_out",
    [
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
        (
            dict(chars="*", maxn=0),
            "Hello, world!!! I wonder....... How are you doing?!?! lololol",
        ),
    ]
)
def test_normalize_repeating_chars(kwargs, text_out):
    text_in = "**Hello**, world!!! I wonder....... How are *you* doing?!?! lololol"
    assert preprocessing.normalize.repeating_chars(text_in, **kwargs) == text_out


@pytest.mark.parametrize(
    "text_in, text_out",
    [
        ("Well… That's a long story.", "Well... That's a long story."),
    ]
)
def test_normalize_unicode(text_in, text_out):
    assert preprocessing.normalize.unicode(text_in, form="NFKC") == text_out


@pytest.mark.parametrize(
    "text_in, text_out",
    [
        ("Hello,  world!", "Hello, world!"),
        ("Hello,     world!", "Hello, world!"),
        ("Hello,\tworld!", "Hello, world!"),
        ("Hello,\t\t  world!", "Hello, world!"),
        ("Hello,\n\nworld!", "Hello,\nworld!"),
        ("Hello,\r\nworld!", "Hello,\nworld!"),
        ("Hello\uFEFF, world!", "Hello, world!"),
        ("Hello\u200B\u200B, world!", "Hello, world!"),
        ("Hello\uFEFF,\n\n\nworld   !  ", "Hello,\nworld !"),
    ]
)
def test_normalize_whitespace(text_in, text_out):
    assert preprocessing.normalize.whitespace(text_in) == text_out
