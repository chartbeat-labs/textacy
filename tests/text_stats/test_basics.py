import pytest

from textacy import extract
from textacy import make_spacy_doc
from textacy.text_stats import basics


@pytest.fixture(scope="module")
def doc():
    text = (
        "Many years later, as he faced the firing squad, Colonel Aureliano Buendía was "
        "to remember that distant afternoon when his father took him to discover ice."
    )
    return make_spacy_doc(text, lang="en")


@pytest.fixture(scope="module")
def words(doc):
    return list(
        extract.words(doc, filter_punct=True, filter_stops=False, filter_nums=False)
    )


def test_n_words(doc, words):
    n_doc = basics.n_words(doc)
    n_words = basics.n_words(words)
    assert n_doc == n_words == pytest.approx(26, rel=0.05)


def test_n_unique_words(doc, words):
    n_doc = basics.n_unique_words(doc)
    n_words = basics.n_unique_words(words)
    assert n_doc == n_words == pytest.approx(25, rel=0.05)


def test_n_chars_per_word(doc, words):
    n_doc = basics.n_chars_per_word(doc)
    n_words = basics.n_chars_per_word(words)
    n_expected = (
        4, 5, 5, 2, 2, 5, 3, 6, 5, 7, 9, 7, 3, 2, 8, 4, 7, 9, 4, 3, 6, 4, 3, 2, 8, 3,
    )
    assert n_doc == n_words == n_expected


def test_n_chars(words):
    n_chars_per_word = basics.n_chars_per_word(words)
    n_chars = basics.n_chars(n_chars_per_word)
    assert n_chars == pytest.approx(126, rel=0.05)


@pytest.mark.parametrize("min_n_chars,n_exp", [(6, 9), (7, 7), (8, 4)])
def test_n_long_words(words, min_n_chars, n_exp):
    n_chars_per_word = basics.n_chars_per_word(words)
    n_long_words = basics.n_long_words(n_chars_per_word, min_n_chars=min_n_chars)
    assert n_long_words == n_exp


# NOTE: you can see how the hack syllable counting stumbles, especially on short words
def test_n_syllables_per_word(doc, words):
    n_doc = basics.n_syllables_per_word(doc, "en")
    n_words = basics.n_syllables_per_word(words, "en")
    n_expected = (
        1, 1, 1, 1, 1, 1, 1, 2, 1, 3, 3, 1, 1, 1, 3, 1, 2, 3, 1, 1, 2, 1, 1, 1, 3, 1
    )
    assert n_doc == n_words == n_expected


def test_n_syllables(words):
    n_syllables_per_word = basics.n_syllables_per_word(words, "en")
    n_syllables = basics.n_syllables(n_syllables_per_word)
    assert n_syllables == pytest.approx(39, rel=0.05)


def test_n_monosyllable_words(words):
    n_syllables_per_word = basics.n_syllables_per_word(words, "en")
    n_monosyllable_words = basics.n_monosyllable_words(n_syllables_per_word)
    assert n_monosyllable_words == pytest.approx(18, rel=0.05)


@pytest.mark.parametrize("min_n_syllables,n_exp", [(2, 8), (3, 5), (4, 0)])
def test_n_polysyllable_words(words, min_n_syllables, n_exp):
    n_syllables_per_word = basics.n_syllables_per_word(words, "en")
    n_polysyllable_words = basics.n_polysyllable_words(
        n_syllables_per_word, min_n_syllables=min_n_syllables,
    )
    assert n_polysyllable_words == n_exp


def test_n_sents(doc):
    n_obs = basics.n_sents(doc)
    assert n_obs == 1


def test_entropy(doc, words):
    val_doc = basics.entropy(doc)
    val_words = basics.entropy(words)
    assert val_doc == pytest.approx(4.623516641218013, abs=1e-4)
    assert val_words == pytest.approx(4.623516641218013, abs=1e-4)
