from __future__ import absolute_import, unicode_literals

import datetime
import os

import pytest

from textacy import compat
from textacy.datasets.imdb import IMDB

DATASET = IMDB()


def _skipif():
    try:
        DATASET._check_data()
        return False
    except OSError:
        return True


pytestmark = pytest.mark.skipif(
    _skipif(),
    reason="IMDB dataset must be downloaded before running tests",
)


@pytest.mark.skip("No need to download a new dataset every time")
def test_download(tmpdir):
    dataset = IMDB(data_dir=str(tmpdir))
    dataset.download()
    assert all(os.path.isfile(filepath) for filepath in dataset.filepaths)


def test_oserror(tmpdir):
    dataset = IMDB(data_dir=str(tmpdir))
    with pytest.raises(OSError):
        _ = list(dataset.texts())


def test_texts():
    texts = list(DATASET.texts(limit=3))
    assert len(texts) > 0
    for text in texts:
        assert isinstance(text, compat.unicode_)


def test_texts_limit():
    for limit in (1, 5, 10):
        assert sum(1 for _ in DATASET.texts(limit=limit)) == limit


def test_texts_min_len():
    for min_len in (100, 200, 500):
        assert all(
            len(text) >= min_len for text in DATASET.texts(min_len=min_len, limit=10)
        )


def test_records():
    for text, meta in DATASET.records(limit=3):
        assert isinstance(text, compat.unicode_)
        assert isinstance(meta, dict)


def test_records_subset():
    subsets = ({"train"}, {"test", "train"})
    for subset in subsets:
        records = list(DATASET.records(subset=subset, limit=10))
        assert all(
            meta["subset"] in subset
            for text, meta in records
        )


def test_records_label():
    labels = ({"pos"}, {"neg", "unsup"})
    for label in labels:
        records = list(DATASET.records(label=label, limit=10))
        assert all(
            meta["label"] in label
            for text, meta in records
        )


def test_records_rating_range():
    rating_ranges = ((6, 8), [8, 10])
    for rating_range in rating_ranges:
        records = list(DATASET.records(rating_range=rating_range, limit=10))
        assert all(
            rating_range[0] <= meta["rating"] < rating_range[1]
            for text, meta in records
        )


def test_bad_filters():
    bad_filters = (
        {"min_len": -1},
    )
    for bad_filter in bad_filters:
        with pytest.raises(ValueError):
            list(DATASET.texts(**bad_filter))
    bad_filters = (
        {"rating_range": 5},
        {"rating_range": ("1", "10")},
    )
    for bad_filter in bad_filters:
        with pytest.raises(TypeError):
            list(DATASET.texts(**bad_filter))
