from __future__ import absolute_import, unicode_literals

import pytest
import spacy

from textacy import cache


def test_cache_clear():
    cache.clear()
    assert len(cache.LRU_CACHE.keys()) == 0


def test_load_spacy_lang():
    for lang in ["en", "en_core_web_sm"]:
        for disable in [None, ("tagger", "parser", "ner")]:
            assert isinstance(
                cache.load_spacy_lang(lang, disable=disable), spacy.language.Language
            )


def test_load_spacy_lang_hashability():
    with pytest.raises(TypeError):
        _ = cache.load_spacy_lang("en", disable=["tagger", "parser", "ner"])


def test_load_pyphen():
    for lang in ("en", "es"):
        _ = cache.load_hyphenator(lang=lang)
        assert True


@pytest.mark.skip(reason="We don't download DepecheMood for tests")
def test_load_depechemood():
    for weighting in ("freq", "normfreq", "tfidf"):
        assert isinstance(cache.load_depechemood(weighting=weighting), dict)


def test_cache_size():
    # check cache size; low thresh but still larger than if the size of
    # loaded data was not being correctly assessed
    # NOTE: must come *after* the previous functions that added data back in
    assert cache.LRU_CACHE.currsize >= 1000
