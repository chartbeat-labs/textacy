import pytest

from textacy import cache


def test_cache_clear():
    cache.clear()
    assert len(cache.LRU_CACHE.keys()) == 0


def test_load_pyphen():
    for lang in ("en", "es"):
        _ = cache.load_hyphenator(lang=lang)
        assert True


def test_cache_size():
    # check cache size; low thresh but still larger than if the size of
    # loaded data was not being correctly assessed
    # NOTE: must come *after* the previous functions that added data back in
    assert cache.LRU_CACHE.currsize >= 1000
