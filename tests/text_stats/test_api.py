import collections
from contextlib import nullcontext as does_not_raise

import pyphen
import pytest
import spacy

import textacy
import textacy.text_stats


@pytest.fixture(scope="module")
def en_doc():
    text = (
        "Many years later, as he faced the firing squad, Colonel Aureliano Buendía was "
        "to remember that distant afternoon when his father took him to discover ice. "
        "At that time Macondo was a village of twenty adobe houses, built on the bank "
        "of a river of clear water that ran along a bed of polished stones, which were "
        "white and enormous, like prehistoric eggs. The world was so recent that many "
        "things lacked names, and in order to indicate them it was necessary to point."
    )
    return textacy.make_spacy_doc(text, lang="en_core_web_sm")


@pytest.fixture(scope="module")
def ts_en(en_doc):
    return textacy.text_stats.TextStats(en_doc)


@pytest.fixture(scope="module")
def ts_es():
    text = (
        "Muchos años después, frente al pelotón de fusilamiento, el coronel Aureliano "
        "Buendía había de recordar aquella tarde remota en que su padre lo llevó a "
        "conocer el hielo. Macondo era entonces una aldea de veinte casas de barro y "
        "cañabrava construidas a la orilla de un río de aguas diáfanas que se precipitaban "
        "por un lecho de piedras pulidas, blancas y enormes como huevos prehistóricos. "
        "El mundo era tan reciente, que muchas cosas carecían de nombre, y para "
        "mencionarlas había que señalarlas con el dedo."
    )
    return textacy.text_stats.TextStats(
        textacy.make_spacy_doc(text, lang="es_core_news_sm")
    )


@pytest.mark.parametrize(
    "lang, attr_name, attr_type, attr_subtype, exp_val",
    [
        ("en", "n_sents", int, None, 3),
        ("en", "n_words", int, None, 84),
        ("en", "n_unique_words", int, None, 66),
        ("en", "n_long_words", int, None, 14),
        ("en", "n_chars_per_word", tuple, int, None),
        ("en", "n_chars", int, None, 372),
        ("en", "n_syllables_per_word", tuple, int, None),
        ("en", "n_syllables", int, None, 111),
        ("en", "n_monosyllable_words", int, None, 66),
        ("en", "n_polysyllable_words", int, None, 8),
        ("en", "entropy", float, None, 5.9071),
        ("es", "n_sents", int, None, 3),
        ("es", "n_words", int, None, 85),
        ("es", "n_unique_words", int, None, 66),
        ("es", "n_long_words", int, None, 24),
        ("es", "n_chars_per_word", tuple, int, None),
        ("es", "n_chars", int, None, 412),
        ("es", "n_syllables_per_word", tuple, int, None),
        ("es", "n_syllables", int, None, 160),
        ("es", "n_monosyllable_words", int, None, 38),
        ("es", "n_polysyllable_words", int, None, 20),
        ("es", "entropy", float, None, 5.8269),
    ],
)
def test_basics_attrs(ts_en, ts_es, lang, attr_name, attr_type, attr_subtype, exp_val):
    # NOTE: this is awkward, and it seems like there should be a better way
    ts = ts_en if lang == "en" else ts_es
    assert hasattr(ts, attr_name)
    obs_val = getattr(ts, attr_name)
    assert isinstance(obs_val, attr_type)
    if attr_subtype is not None:
        assert all(isinstance(ov, attr_subtype) for ov in obs_val)
    if exp_val is not None:
        if attr_type is float:
            assert obs_val == pytest.approx(exp_val, rel=1e-2)
        else:
            assert obs_val == exp_val


@pytest.mark.parametrize(
    "lang, name, exp",
    [
        (
            "en",
            "pos",
            {
                "ADJ": 10,
                "NOUN": 19,
                "ADV": 2,
                "PUNCT": 9,
                "SCONJ": 4,
                "PRON": 7,
                "VERB": 9,
                "DET": 7,
                "PROPN": 5,
                "AUX": 5,
                "PART": 4,
                "ADP": 9,
                "NUM": 1,
                "CCONJ": 2,
            },
        ),
        (
            "en",
            "tag",
            {
                "JJ": 10,
                "NNS": 6,
                "RB": 2,
                ",": 6,
                "IN": 12,
                "PRP": 4,
                "VBD": 9,
                "DT": 7,
                "NN": 13,
                "NNP": 5,
                "TO": 4,
                "VB": 4,
                "WRB": 1,
                "PRP$": 1,
                ".": 3,
                "CD": 1,
                "VBN": 1,
                "WDT": 2,
                "CC": 2,
            },
        ),
        (
            "en",
            "dep",
            {
                "amod": 6,
                "npadvmod": 1,
                "advmod": 3,
                "punct": 9,
                "mark": 2,
                "nsubj": 9,
                "advcl": 1,
                "det": 8,
                "compound": 4,
                "dobj": 6,
                "ROOT": 3,
                "aux": 4,
                "xcomp": 3,
                "poss": 1,
                "relcl": 3,
                "prep": 8,
                "pobj": 9,
                "attr": 1,
                "nummod": 1,
                "acl": 2,
                "acomp": 3,
                "cc": 2,
                "conj": 2,
                "ccomp": 2,
            },
        ),
        (
            "en",
            "morph",
            {
                "Degree": {"Pos": 10},
                "Number": {"Plur": 7, "Sing": 27},
                "PunctType": {"Comm": 6, "Peri": 3},
                "Case": {"Nom": 2, "Acc": 2},
                "Gender": {"Masc": 3, "Neut": 1},
                "Person": {"3": 9},
                "PronType": {"Prs": 5, "Art": 6, "Dem": 1, "Rel": 1},
                "Tense": {"Past": 10},
                "VerbForm": {"Fin": 9, "Inf": 4, "Part": 1},
                "Definite": {"Def": 3, "Ind": 3},
                "Mood": {"Ind": 5},
                "Poss": {"Yes": 1},
                "NumType": {"Card": 1},
                "Aspect": {"Perf": 1},
                "ConjType": {"Cmp": 2},
            },
        ),
    ],
)
def test_counts_method(ts_en, ts_es, lang, name, exp):
    # the expected values are very sensitive to spacy model's token annotations
    # so we try to assess "more or less" equal counts
    ts = ts_en if lang == "en" else ts_es
    obs = ts.counts(name)
    assert isinstance(obs, dict)
    if name != "morph":
        assert all(isinstance(val, int) for val in obs.values())
        diffs = collections.Counter(ts.counts(name))
        diffs.subtract(exp)
        assert all(abs(val) <= 1 for val in diffs.values())
    else:
        assert all(isinstance(val_counts, dict) for val_counts in obs.values())
        assert all(
            isinstance(val, int)
            for val_counts in obs.values()
            for val in val_counts.values()
        )
        assert list(obs.keys()) == list(exp.keys())


@pytest.mark.parametrize(
    "lang, method_name, exp_val",
    [
        ("en", "automated-readability-index", 13.42857),
        ("en", "automatic-arabic-readability-index", 1261.21),
        ("en", "coleman-liau-index", 9.1818),
        ("en", "flesch-kincaid-grade-level", 10.92285),
        ("en", "flesch-reading-ease", 66.6221),
        ("en", "gulpease-index", 55.4285),
        ("en", "gunning-fog-index", 15.00952),
        ("en", "lix", 44.6666),
        ("en", "mu-legibility-index", 97.236),
        ("en", "perspicuity-index", 96.5100),
        ("en", "smog-index", 12.45797),
        ("en", "wiener-sachtextformel", 5.2418),
        ("es", "automated-readability-index", 15.56631),
        ("es", "automatic-arabic-readability-index", 1393.424),
        ("es", "coleman-liau-index", 11.6549),
        ("es", "flesch-kincaid-grade-level", 17.6717),
        ("es", "flesch-reading-ease", 64.9938),
        ("es", "gulpease-index", 51.1176),
        ("es", "gunning-fog-index", 20.74509),
        ("es", "lix", 56.5686),
        ("es", "mu-legibility-index", 58.2734),
        ("es", "perspicuity-index", 61.2310),
        ("es", "smog-index", 17.87934),
        ("es", "wiener-sachtextformel", 10.6155),
    ],
)
def test_readability_method(ts_en, ts_es, lang, method_name, exp_val):
    ts = ts_en if lang == "en" else ts_es
    assert ts.readability(method_name) == pytest.approx(exp_val, rel=0.05)


@pytest.mark.parametrize(
    "method_name, kwargs, exp_val",
    [
        ("ttr", {}, 0.785),
        ("ttr", {"variant": "root"}, 7.202),
        ("log-ttr", {}, 0.946),
        ("log-ttr", {"variant": "dugast"}, 35.354),
        ("segmented-ttr", {}, 0.840),
        ("segmented-ttr", {"variant": "moving-avg"}, 0.820),
        ("segmented-ttr", {"variant": "mean", "segment_size": 25}, 0.920),
        ("mtld", {}, 109.759),
        ("mtld", {"min_ttr": 0.75}, 98.0),
        ("hdd", {}, 0.858),
        ("hdd", {"sample_size": 50}, 0.840),
    ],
)
def test_diversity_method(ts_en, method_name, kwargs, exp_val):
    assert ts_en.diversity(method_name, **kwargs) == pytest.approx(exp_val, rel=0.05)


@pytest.mark.parametrize(
    "lang, context",
    [
        ("en", does_not_raise()),
        ("es", does_not_raise()),
        ("un", pytest.raises(KeyError)),
    ],
)
def test_load_hyphenator(lang, context):
    with context:
        hyphenator = textacy.text_stats.utils.load_hyphenator(lang=lang)
        assert isinstance(hyphenator, pyphen.Pyphen)


def test_get_set_doc_extensions(en_doc):
    for name in textacy.text_stats.get_doc_extensions().keys():
        assert en_doc.has_extension(name) is False
    textacy.text_stats.set_doc_extensions(force=True)
    for name in textacy.text_stats.get_doc_extensions().keys():
        assert en_doc.has_extension(name)


@pytest.mark.parametrize(
    "ext_name, func",
    [
        ("n_sents", textacy.text_stats.basics.n_sents),
        ("n_words", textacy.text_stats.basics.n_words),
        ("n_chars", textacy.text_stats.basics.n_chars),
        ("n_syllables", textacy.text_stats.basics.n_syllables),
    ],
)
def test_check_doc_extensions(en_doc, ext_name, func):
    assert getattr(en_doc._, ext_name) == func(en_doc)


def test_remove_doc_extensions(en_doc):
    textacy.text_stats.remove_doc_extensions()
    for name in textacy.text_stats.get_doc_extensions().keys():
        assert en_doc.has_extension(name) is False
