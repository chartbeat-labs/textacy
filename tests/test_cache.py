from __future__ import absolute_import, unicode_literals

import pytest
import spacy

from textacy import cache


def test_load_spacy():
    for lang in ("en", "en_core_web_sm"):
        for disable in (None, ("parser", "ner")):
            assert isinstance(
                cache.load_spacy(lang, disable=disable), spacy.language.Language
            )


def test_load_spacy_hashability():
    with pytest.raises(TypeError):
        _ = cache.load_spacy("en", disable=["tagger", "parser", "ner"])


def test_load_pyphen():
    for lang in ("en", "es"):
        _ = cache.load_hyphenator(lang=lang)
        assert True


@pytest.mark.skip(reason="We don't download DepecheMood for tests")
def test_load_depechemood():
    for weighting in ("freq", "normfreq", "tfidf"):
        assert isinstance(cache.load_depechemood(weighting=weighting), dict)


def test_cache_clear():
    cache.clear()
    _ = cache.load_hyphenator(lang="en")
    _ = cache.load_spacy("en")
    assert len(cache.LRU_CACHE.keys()) >= 2
    # check cache size; low thresh but still larger than if the size of
    # loaded data was not being correctly assessed
    assert cache.LRU_CACHE.currsize >= 1000
