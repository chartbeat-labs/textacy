import pytest

from textacy import lang_utils


@pytest.mark.parametrize(
    "text,lang",
    [
        ("This is a short example sentence in English.", "en"),
        ("Esta es una breve oración de ejemplo en español.", "es"),
        ("Ceci est un exemple court phrase en français.", "fr"),
        ("1", "un"),
        (" ", "un"),
        ("", "un"),
    ]
)
class TestLangIdentifier:

    def test_module_alias(self, text, lang):
        assert lang_utils.identify_lang(text) == lang

    def test_identify_topn_langs(self, text, lang):
        topn_langs = lang_utils.lang_identifier.identify_topn_langs(text)
        assert topn_langs[0][0] == lang
        assert topn_langs[0][1] >= 0.5

    def test_identify_topn_langs_topn(self, text, lang):
        for n in [1, 2, 3]:
            topn_langs = lang_utils.lang_identifier.identify_topn_langs(text, topn=n)
            assert isinstance(topn_langs, list)
            assert len(topn_langs) == n or lang == "un"
