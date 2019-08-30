import pytest

from textacy import load_spacy_lang
from textacy import cache


def test_cache_clear():
    cache.clear()
    assert len(cache.LRU_CACHE.keys()) == 0


def test_cache_size():
    # check cache size; low thresh but still larger than if the size of
    # loaded data was not being correctly assessed
    # NOTE: should come *after* the function that clears the cache
    _ = load_spacy_lang("en")
    assert cache.LRU_CACHE.currsize >= 1000
