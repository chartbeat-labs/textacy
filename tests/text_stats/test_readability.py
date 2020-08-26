from contextlib import ExitStack as does_not_raise
# TODO: when only supporting PY3.7+, use this instead
# from contextlib import nullcontext as does_not_raise

import pytest

from textacy.text_stats import readability


@pytest.fixture(scope="module")
def basics():
    return dict(
        n_sents=3,
        n_words=84,
        n_unique_words=66,
        n_chars=372,
        n_long_words=14,
        n_syllables=113,
        n_monosyllable_words=63,
        n_polysyllable_words=8,
    )


@pytest.mark.parametrize(
    "n_chars, n_words, n_sents, exp",
    [
        (372, 84, 3, 13.4285),
        (200, 100, 2, 12.99),
        (1, 1, 1, -16.22),
    ]
)
def test_automated_readability_index(n_chars, n_words, n_sents, exp):
    obs = readability.automated_readability_index(n_chars, n_words, n_sents)
    assert obs == pytest.approx(exp, rel=1e-2)


@pytest.mark.parametrize(
    "n_chars, n_words, n_sents, exp",
    [
        (372, 84, 3, 1261.2128),
        (200, 100, 2, 720.86),
        (1, 1, 1, 5.95),
    ]
)
def test_automatic_arabic_readability_index(n_chars, n_words, n_sents, exp):
    obs = readability.automatic_arabic_readability_index(n_chars, n_words, n_sents)
    assert obs == pytest.approx(exp, rel=1e-2)


@pytest.mark.parametrize(
    "n_chars, n_words, n_sents, exp",
    [
        (372, 84, 3, 9.1818),
        (200, 100, 2, -4.6328),
        (1, 1, 1, -39.5082),
    ]
)
def test_coleman_liau_index(n_chars, n_words, n_sents, exp):
    obs = readability.coleman_liau_index(n_chars, n_words, n_sents)
    assert obs == pytest.approx(exp, rel=1e-2)


@pytest.mark.parametrize(
    "n_syllables, n_words, n_sents, exp",
    [
        (113, 84, 3, 11.2038),
        (200, 100, 2, 27.51),
        (1, 1, 1, -3.39999),
    ]
)
def test_flesch_kincaid_grade_level(n_syllables, n_words, n_sents, exp):
    obs = readability.flesch_kincaid_grade_level(n_syllables, n_words, n_sents)
    assert obs == pytest.approx(exp, rel=1e-2)


@pytest.mark.parametrize(
    "n_syllables, n_words, n_sents, lang, exp",
    [
        (113, 84, 3, "en", 64.6078),
        (200, 100, 2, "en", -13.1149),
        (1, 1, 1, "en", 121.22),
    ]
)
def test_flesch_reading_ease(n_syllables, n_words, n_sents, lang, exp):
    obs = readability.flesch_reading_ease(n_syllables, n_words, n_sents, lang=lang)
    assert obs == pytest.approx(exp, rel=1e-2)


@pytest.mark.parametrize(
    "lang,context",
    [
        (None, does_not_raise()),
        ("en", does_not_raise()),
        ("de", does_not_raise()),
        ("es", does_not_raise()),
        ("fr", does_not_raise()),
        ("it", does_not_raise()),
        ("nl", does_not_raise()),
        ("pt", does_not_raise()),
        ("ru", does_not_raise()),
        ("tr", does_not_raise()),
        ("ja", pytest.raises(ValueError)),
        ("un", pytest.raises(ValueError)),
        ("zh", pytest.raises(ValueError)),
    ],
)
def test_flesch_reading_ease_lang(lang, context):
    with context:
        _ = readability.flesch_reading_ease(100, 50, 2, lang=lang)


@pytest.mark.parametrize(
    "n_chars, n_words, n_sents, exp",
    [
        (372, 84, 3, 55.4285),
        (200, 100, 2, 75.0),
        (1, 1, 1, 379.0),
    ]
)
def test_gulpease_index(n_chars, n_words, n_sents, exp):
    obs = readability.gulpease_index(n_chars, n_words, n_sents)
    assert obs == pytest.approx(exp, rel=1e-2)


@pytest.mark.parametrize(
    "n_words, n_polysyllable_words, n_sents, exp",
    [
        (84, 8, 3, 15.0095),
        (50, 10, 2, 18.0),
        (1, 1, 1, 40.40),
    ]
)
def test_gunning_fog_index(n_words, n_polysyllable_words, n_sents, exp):
    obs = readability.gunning_fog_index(n_words, n_polysyllable_words, n_sents)
    assert obs == pytest.approx(exp, rel=1e-2)


@pytest.mark.parametrize(
    "n_words, n_long_words, n_sents, exp",
    [
        (84, 14, 3, 44.6666),
        (50, 5, 2, 35.0),
        (1, 1, 1, 101.0),
    ]
)
def test_lix(n_words, n_long_words, n_sents, exp):
    obs = readability.lix(n_words, n_long_words, n_sents)
    assert obs == pytest.approx(exp, rel=1e-2)


@pytest.mark.parametrize(
    "n_chars_per_word, exp",
    [
        [(4, 5, 5, 2, 2, 5, 3, 6, 5, 7), 180.3278],
        [(6, 4, 7, 6, 2, 7, 2, 12, 2, 7), 62.1468],
        [(1, 2, 1, 2, 1, 2, 1, 2, 1, 2), 600.0],
    ]
)
def test_mu_legibility_index(n_chars_per_word, exp):
    obs = readability.mu_legibility_index(n_chars_per_word)
    assert obs == pytest.approx(exp, rel=1e-2)


@pytest.mark.parametrize(
    "n_syllables, n_words, n_sents, exp",
    [
        (113, 84, 3, 95.02666),
        (200, 100, 2, 32.2350),
        (1, 1, 1, 143.5350),
    ]
)
def test_perspicuity_index(n_syllables, n_words, n_sents, exp):
    obs = readability.perspicuity_index(n_syllables, n_words, n_sents)
    assert obs == pytest.approx(exp, rel=1e-2)


@pytest.mark.parametrize(
    "n_polysyllable_words, n_sents, exp",
    [
        (8, 3, 12.4579),
        (10, 2, 15.9031),
        (1, 1, 8.8418),
    ]
)
def test_smog_index(n_polysyllable_words, n_sents, exp):
    obs = readability.smog_index(n_polysyllable_words, n_sents)
    assert obs == pytest.approx(exp, rel=1e-2)


@pytest.mark.parametrize(
    "n_words, n_polysyllable, n_monosyllable, n_long, n_sents, variant, exp",
    [
        (84, 8, 63, 14, 3, 1, 5.3586),
        (84, 8, 63, 14, 3, 2, 6.1303),
        (84, 8, 63, 14, 3, 3, 7.0415),
        (84, 8, 63, 14, 3, 4, 8.3571),
        (1, 1, 1, 1, 1, 1, 28.3422),
    ]
)
def test_wiener_sachtextformel(
    n_words, n_polysyllable, n_monosyllable, n_long, n_sents, variant, exp,
):
    obs = readability.wiener_sachtextformel(
        n_words, n_polysyllable, n_monosyllable, n_long, n_sents, variant=variant,
    )
    assert obs == pytest.approx(exp, rel=1e-2)


@pytest.mark.parametrize(
    "variant, context",
    [
        (0, pytest.raises(ValueError)),
        (1, does_not_raise()),
        (2, does_not_raise()),
        (3, does_not_raise()),
        (4, does_not_raise()),
        (5, pytest.raises(ValueError)),
    ]
)
def test_wiener_sachtextformel_variant(variant, context):
    with context:
        _ = readability.wiener_sachtextformel(84, 8, 63, 14, 3, variant=variant)
