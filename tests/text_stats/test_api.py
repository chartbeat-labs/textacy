from contextlib import ExitStack as does_not_raise
# TODO: when only supporting PY3.7+, use this instead
# from contextlib import nullcontext as does_not_raise

import pyphen
import pytest

import textacy
import textacy.text_stats


@pytest.fixture(scope="module")
def ts_en():
    text = (
        "Many years later, as he faced the firing squad, Colonel Aureliano Buendía was "
        "to remember that distant afternoon when his father took him to discover ice. "
        "At that time Macondo was a village of twenty adobe houses, built on the bank "
        "of a river of clear water that ran along a bed of polished stones, which were "
        "white and enormous, like prehistoric eggs. The world was so recent that many "
        "things lacked names, and in order to indicate them it was necessary to point."
    )
    return textacy.TextStats(textacy.make_spacy_doc(text, lang="en_core_web_sm"))


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
    return textacy.TextStats(textacy.make_spacy_doc(text, lang="es"))


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
    ]
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
    "lang, attr_name, exp_val",
    [
        ("en", "automated_readability_index", 13.42857),
        ("en", "automatic_arabic_readability_index", 1261.21),
        ("en", "coleman_liau_index", 9.1818),
        ("en", "flesch_kincaid_grade_level", 10.92285),
        ("en", "flesch_reading_ease", 66.6221),
        ("en", "gulpease_index", 55.4285),
        ("en", "gunning_fog_index", 15.00952),
        ("en", "lix", 44.6666),
        ("en", "mu_legibility_index", 97.236),
        ("en", "perspicuity_index", 96.5100),
        ("en", "smog_index", 12.45797),
        ("en", "wiener_sachtextformel", 5.2418),
        ("es", "automated_readability_index", 15.56631),
        ("es", "automatic_arabic_readability_index", 1393.424),
        ("es", "coleman_liau_index", 11.6549),
        ("es", "flesch_kincaid_grade_level", 17.6717),
        ("es", "flesch_reading_ease", 64.9938),
        ("es", "gulpease_index", 51.1176),
        ("es", "gunning_fog_index", 20.74509),
        ("es", "lix", 56.5686),
        ("es", "mu_legibility_index", 58.2734),
        ("es", "perspicuity_index", 61.2310),
        ("es", "smog_index", 17.87934),
        ("es", "wiener_sachtextformel", 10.6155),
    ]
)
def test_readability_attrs(ts_en, ts_es, lang, attr_name, exp_val):
    # NOTE: this is awkward, and it seems like there should be a better way
    ts = ts_en if lang == "en" else ts_es
    assert hasattr(ts, attr_name)
    assert getattr(ts, attr_name) == pytest.approx(exp_val, rel=0.05)


@pytest.mark.parametrize(
    "lang, context",
    [
        ("en", does_not_raise()),
        ("es", does_not_raise()),
        ("un", pytest.raises(KeyError)),
    ]
)
def test_load_hyphenator(lang, context):
    with context:
        hyphenator = textacy.text_stats.load_hyphenator(lang=lang)
        assert isinstance(hyphenator, pyphen.Pyphen)
