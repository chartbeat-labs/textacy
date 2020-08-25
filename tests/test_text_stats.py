from contextlib import ExitStack as does_not_raise
# TODO: when only supporting PY3.7+, use this instead
# from contextlib import nullcontext as does_not_raise

import pytest

from textacy import text_stats
from textacy import make_spacy_doc


@pytest.fixture(scope="module")
def ts():
    text = """
    Mr. Speaker, 480,000 Federal employees are working without pay, a form of involuntary servitude; 280,000 Federal employees are not working, and they will be paid. Virtually all of these workers have mortgages to pay, children to feed, and financial obligations to meet.
    Mr. Speaker, what is happening to these workers is immoral, is wrong, and must be rectified immediately. Newt Gingrich and the Republican leadership must not continue to hold the House and the American people hostage while they push their disastrous 7-year balanced budget plan. The gentleman from Georgia, Mr. Gingrich, and the Republican leadership must join Senator Dole and the entire Senate and pass a continuing resolution now, now to reopen Government.
    Mr. Speaker, that is what the American people want, that is what they need, and that is what this body must do.
    """.strip()
    doc = make_spacy_doc(text, lang="en")
    ts_ = text_stats.TextStats(doc)
    return ts_


def test_n_sents(ts):
    assert ts.n_sents == 6


def test_n_words(ts):
    assert ts.n_words == 136


def test_n_chars(ts):
    assert ts.n_chars == 685


def test_n_syllables(ts):
    assert ts.n_syllables == 214


def test_n_unique_words(ts):
    assert ts.n_unique_words == 80


def test_n_long_words(ts):
    assert ts.n_long_words == 43


def test_n_monosyllable_words(ts):
    assert ts.n_monosyllable_words == 90


def test_n_polysyllable_words(ts):
    assert ts.n_polysyllable_words == 24


def test_flesch_kincaid_grade_level(ts):
    assert ts.flesch_kincaid_grade_level == pytest.approx(11.817647058823532, rel=1e-2)


def test_flesch_reading_ease(ts):
    assert ts.flesch_reading_ease == pytest.approx(50.707745098039254, rel=1e-2)


def test_smog_index(ts):
    assert ts.smog_index == pytest.approx(14.554592549557764, rel=1e-2)


def test_gunning_fog_index(ts):
    assert ts.gunning_fog_index == pytest.approx(16.12549019607843, rel=1e-2)


def test_coleman_liau_index(ts):
    assert ts.coleman_liau_index == pytest.approx(12.509300816176474, rel=1e-2)


def test_automated_readability_index(ts):
    assert ts.automated_readability_index == pytest.approx(13.626495098039214, rel=1e-2)


def test_lix(ts):
    assert ts.lix == pytest.approx(54.28431372549019, rel=1e-2)


def test_gulpease_index(ts):
    assert ts.gulpease_index == pytest.approx(51.86764705882353, rel=1e-2)


def test_wiener_sachtextformel(ts):
    assert ts.wiener_sachtextformel == pytest.approx(8.266410784313727, rel=1e-2)


def test_wiener_sachtextformel_variant1(ts):
    assert ts.wiener_sachtextformel == text_stats.readability.wiener_sachtextformel(
        ts.n_words,
        ts.n_polysyllable_words,
        ts.n_monosyllable_words,
        ts.n_long_words,
        ts.n_sents,
        variant=1,
    )
    assert text_stats.readability.wiener_sachtextformel(
        ts.n_words,
        ts.n_polysyllable_words,
        ts.n_monosyllable_words,
        ts.n_long_words,
        ts.n_sents,
        variant=1,
    ) == pytest.approx(8.266410784313727, rel=1e-2)


def test_wiener_sachtextformel_variant2(ts):
    assert text_stats.readability.wiener_sachtextformel(
        ts.n_words,
        ts.n_polysyllable_words,
        ts.n_monosyllable_words,
        ts.n_long_words,
        ts.n_sents,
        variant=2,
    ) == pytest.approx(8.916400980392158, rel=1e-2)


def test_wiener_sachtextformel_variant3(ts):
    assert text_stats.readability.wiener_sachtextformel(
        ts.n_words,
        ts.n_polysyllable_words,
        ts.n_monosyllable_words,
        ts.n_long_words,
        ts.n_sents,
        variant=3,
    ) == pytest.approx(8.432423529411766, rel=1e-2)


def test_wiener_sachtextformel_variant4(ts):
    assert text_stats.readability.wiener_sachtextformel(
        ts.n_words,
        ts.n_polysyllable_words,
        ts.n_monosyllable_words,
        ts.n_long_words,
        ts.n_sents,
        variant=4,
    ) == pytest.approx(9.169619607843138, rel=1e-2)


@pytest.mark.parametrize(
    "lang,fre,context",
    [
        (None, 50.707745098039254, does_not_raise()),
        ("en", 50.707745098039254, does_not_raise()),
        ("de", 65.28186274509805, does_not_raise()),
        ("es", 89.30823529411765, does_not_raise()),
        ("fr", 68.18156862745099, does_not_raise()),
        ("it", 93.12156862745098, does_not_raise()),
        ("nl", 64.59823529411764, does_not_raise()),
        ("pt", 92.70774509803925, does_not_raise()),
        ("ru", 82.79921568627452, does_not_raise()),
        ("ja", None, pytest.raises(ValueError)),
        ("un", None, pytest.raises(ValueError)),
        ("zh", None, pytest.raises(ValueError)),
    ],
)
def test_flesch_reading_ease_lang(ts, lang, fre, context):
    with context:
        result = text_stats.readability.flesch_reading_ease(
            ts.n_syllables, ts.n_words, ts.n_sents, lang=lang
        )
        assert result == pytest.approx(fre, rel=1e-2)


@pytest.mark.parametrize("lang", ["en", "es"])
def test_load_hyphenator(lang):
    assert text_stats.load_hyphenator(lang=lang)
