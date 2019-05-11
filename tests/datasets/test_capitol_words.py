from __future__ import absolute_import, unicode_literals

import datetime
import os

import pytest

from textacy import compat
from textacy.datasets.capitol_words import CapitolWords

DATASET = CapitolWords()

pytestmark = pytest.mark.skipif(
    DATASET.filepath is None,
    reason="CapitolWords dataset must be downloaded before running tests",
)


@pytest.mark.skip("No need to download a new dataset every time")
def test_download(tmpdir):
    dataset = CapitolWords(data_dir=str(tempdir))
    dataset.download()
    assert os.path.isfile(dataset._filepath)


def test_oserror(tmpdir):
    dataset = CapitolWords(data_dir=str(tmpdir))
    with pytest.raises(OSError):
        _ = list(dataset.texts())


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


def test_records_speaker_name():
    speaker_names = ({"Bernie Sanders"}, {"Joseph Biden", "Rick Santorum"})
    for speaker_name in speaker_names:
        records = list(DATASET.records(speaker_name=speaker_name, limit=3))
        assert len(records) >= 1
        assert all(
            meta["speaker_name"] in speaker_name
            for text, meta in records
        )


def test_records_speaker_party():
    speaker_parties = ({"R"}, {"D", "I"})
    for speaker_party in speaker_parties:
        records = list(DATASET.records(speaker_party=speaker_party, limit=10))
        assert len(records) >= 1
        assert all(
            meta["speaker_party"] in speaker_party
            for text, meta in records
        )


def test_records_chamber():
    chambers = ({"House"}, {"House", "Senate"})
    for chamber in chambers:
        records = list(DATASET.records(chamber=chamber, limit=10))
        assert len(records) >= 1
        assert all(
            meta["chamber"] in chamber
            for text, meta in records
        )


def test_records_congress():
    congresses = ({104}, {104, 114})
    for congress in congresses:
        records = list(DATASET.records(congress=congress, limit=10))
        assert len(records) >= 1
        assert all(
            meta["congress"] in congress
            for text, meta in records
        )


def test_records_date_range():
    date_ranges = (["1997-01-01", "1998-01-01"], ("1997-01-01", "1997-02-01"))
    for date_range in date_ranges:
        records = list(DATASET.records(date_range=date_range, limit=3))
        assert len(records) >= 1
        assert all(
            date_range[0] <= meta["date"] < date_range[1]
            for text, meta in records
        )


def test_bad_filters():
    bad_filters = (
        {"speaker_name": "Burton DeWilde"},
        {"speaker_party": "Whigs"},
        {"chamber": "White House"},
        {"congress": 42},
        {"min_len": -1},
    )
    for bad_filter in bad_filters:
        with pytest.raises(ValueError):
            list(DATASET.texts(**bad_filter))
    bad_filters = (
        {"score_range": ["low", "high"]},
        {"date_range": "2016-01-01"},
        {"date_range": (datetime.date(2000, 1, 1), datetime.date(2001, 1, 1))},
    )
    for bad_filter in bad_filters:
        with pytest.raises(TypeError):
            list(DATASET.texts(**bad_filter))
