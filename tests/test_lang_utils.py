from textacy import lang_utils

GOOD_LANG_SENTS = [
    ("en", "This is a short example sentence in English."),
    ("es", "Esta es una breve oración de ejemplo en español."),
    ("fr", "Ceci est un exemple court phrase en français."),
]

BAD_LANG_SENTS = [
    ("un", "1"),
    ("un", " "),
    ("un", ""),
]


def test_identify_lang():
    for lang, sent in GOOD_LANG_SENTS + BAD_LANG_SENTS:
        assert lang_utils.identify_lang(sent) == lang


def test_identify_topn_langs():
    for lang, sent in GOOD_LANG_SENTS + BAD_LANG_SENTS:
        topn_langs = lang_utils.lang_identifier.identify_topn_langs(sent)
        assert topn_langs[0][0] == lang
        assert topn_langs[0][1] >= 0.5


def test_identify_topn_langs_topn():
    for n in [1, 2, 3]:
        topn_langs = lang_utils.lang_identifier.identify_topn_langs(
            GOOD_LANG_SENTS[0][1], topn=n)
        assert isinstance(topn_langs, list)
        assert len(topn_langs) == n
