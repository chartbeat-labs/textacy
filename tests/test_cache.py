from __future__ import absolute_import, unicode_literals

import pytest
import spacy

from textacy import cache


def test_cache_clear():
    cache.clear()
    assert len(cache.LRU_CACHE.keys()) == 0


class TestLoadSpacyLang(object):

    def test_load_model(self):
        for lang in ["en", "en_core_web_sm"]:
            for disable in [None, ("tagger", "parser", "ner")]:
                assert isinstance(
                    cache.load_spacy_lang(lang, disable=disable),
                    spacy.language.Language
                )

    def test_load_blank(self):
        assert isinstance(
            cache.load_spacy_lang("ar", allow_blank=True),
            spacy.language.Language
        )

    def test_disable_hashability(self):
        with pytest.raises(TypeError):
            _ = cache.load_spacy_lang("en", disable=["tagger", "parser", "ner"])

    def test_bad_name(self):
        for name in ("unk", "un"):
            with pytest.raises((OSError, IOError)):
                _ = cache.load_spacy_lang(name)
        with pytest.raises(ImportError):
            _ = cache.load_spacy_lang("un", allow_blank=True)


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
