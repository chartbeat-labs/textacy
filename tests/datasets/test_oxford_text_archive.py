from __future__ import absolute_import, unicode_literals

import datetime
import os

import pytest

from textacy import compat
from textacy.datasets.oxford_text_archive import OxfordTextArchive

DATASET = OxfordTextArchive()

pytestmark = pytest.mark.skipif(
    DATASET.metadata is None,
    reason="OxfordTextArchive dataset must be downloaded before running tests",
)


@pytest.mark.skip("No need to download a new dataset every time")
def test_download(tmpdir):
    dataset = OxfordTextArchive(data_dir=str(tempdir))
    dataset.download()
    assert os.path.isfile(dataset._metadata_filepath)
    assert os.path.isdir(dataset._text_dirpath)


def test_oserror(tmpdir):
    dataset = OxfordTextArchive(data_dir=str(tmpdir))
    with pytest.raises(OSError):
        _ = list(dataset.texts())


def test_metadata():
    assert DATASET.metadata is not None
    for record in DATASET.records(limit=10):
        pass
    assert all(entry.get("id") for entry in DATASET.metadata.values())
    assert not any(entry.get("text") for entry in DATASET.metadata.values())


def test_texts():
    for text in DATASET.texts(limit=3):
        assert isinstance(text, compat.unicode_)


def test_texts_limit():
    for limit in (1, 5, 10):
        assert sum(1 for _ in DATASET.texts(limit=limit)) == limit


def test_texts_min_len():
    for min_len in (100, 200, 1000):
        assert all(
            len(text) >= min_len for text in DATASET.texts(min_len=min_len, limit=10)
        )


def test_records():
    for text, meta in DATASET.records(limit=3):
        assert isinstance(text, compat.unicode_)
        assert isinstance(meta, dict)


def test_records_author():
    authors = ({"London, Jack"}, {"Burroughs, Edgar Rice", "Wells, H.G. (Herbert George)"})
    for author in authors:
        records = list(DATASET.records(author=author, limit=3))
        assert len(records) >= 1
        assert any(
            athr in author
            for text, meta in records
            for athr in meta["author"]
        )


def test_records_date_range():
    date_ranges = (["1840-01-01", "1860-01-01"], ("1500-01-01", "1600-01-01"))
    for date_range in date_ranges:
        records = list(DATASET.records(date_range=date_range, limit=3))
        assert len(records) >= 1
        assert all(
            date_range[0] <= meta["year"] < date_range[1]
            for text, meta in records
        )


def test_bad_filters():
    bad_filters = (
        {"author": "Burton DeWilde"},
        {"min_len": -1},
    )
    for bad_filter in bad_filters:
        with pytest.raises(ValueError):
            list(DATASET.texts(**bad_filter))
    bad_filters = (
        {"date_range": "2016-01-01"},
        {"date_range": (datetime.date(1800, 1, 1), datetime.date(1900, 1, 1))},
    )
    for bad_filter in bad_filters:
        with pytest.raises(TypeError):
            list(DATASET.texts(**bad_filter))
