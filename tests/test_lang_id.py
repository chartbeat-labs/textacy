import pytest

from textacy.lang_id import identify_lang


@pytest.mark.parametrize(
    "text, exp",
    [
        ("This is a short example sentence in English.", "en"),
        ("Esta es una breve oración de ejemplo en español.", "es"),
        ("Ceci est un exemple court phrase en français.", "fr"),
        ("1", "un"),
        (" ", "un"),
        ("", "un"),
    ]
)
def test_lang_id(text, exp):
    obs = identify_lang(text)
    assert obs == exp
