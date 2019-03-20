from __future__ import absolute_import, unicode_literals

import os

import pytest

from textacy import compat
from textacy.datasets.wikipedia import Wikipedia

DATASET = Wikipedia(lang="en", version="latest")

pytestmark = pytest.mark.skipif(
    DATASET.filename is None,
    reason="Wikipedia dataset must be downloaded before running tests",
)


@pytest.mark.skip("No need to download a new dataset every time")
def test_download(tmpdir):
    dataset = Wikipedia(data_dir=str(tmpdir))
    dataset.download()
    assert os.path.exists(dataset.filename)


def test_ioerror(tmpdir):
    dataset = Wikipedia(data_dir=str(tmpdir))
    with pytest.raises(IOError):
        _ = list(dataset.texts())


def test_texts():
    for text in DATASET.texts(limit=3):
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
    for record in DATASET.records(limit=3):
        assert isinstance(record, dict)


def test_records_fast():
    for record in DATASET.records(limit=3, fast=True):
        assert isinstance(record, dict)


# TODO: test individual parsing functions
