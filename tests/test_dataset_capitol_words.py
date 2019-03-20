from __future__ import absolute_import, unicode_literals

import os

import pytest

from textacy import compat
from textacy.datasets.capitol_words import CapitolWords

DATASET = CapitolWords()

pytestmark = pytest.mark.skipif(
    DATASET.filename is None,
    reason="CapitolWords dataset must be downloaded before running tests",
)


@pytest.mark.skip("No need to download a new dataset every time")
def test_download(tmpdir):
    dataset = CapitolWords(data_dir=str(tempdir))
    dataset.download()
    assert os.path.exists(dataset.filename)


def test_ioerror(tmpdir):
    dataset = CapitolWords(data_dir=str(tmpdir))
    with pytest.raises(IOError):
        _ = list(dataset.texts())


def test_texts():
    for text in DATASET.texts(limit=3):
        assert isinstance(text, compat.unicode_)


def test_texts_limit():
    for limit in (1, 5, 100):
        assert sum(1 for _ in DATASET.texts(limit=limit)) == limit


def test_texts_min_len():
    for min_len in (100, 200, 1000):
        assert all(
            len(text) >= min_len for text in DATASET.texts(min_len=min_len, limit=10)
        )


def test_records():
    for record in DATASET.records(limit=3):
        assert isinstance(record, dict)


def test_records_speaker_name():
    speaker_names = ({"Bernie Sanders"}, {"Ted Cruz", "Barack Obama"})
    for speaker_name in speaker_names:
        assert all(
            r["speaker_name"] in speaker_name
            for r in DATASET.records(speaker_name=speaker_name, limit=10)
        )


def test_records_speaker_party():
    speaker_parties = ({"R"}, {"D", "I"})
    for speaker_party in speaker_parties:
        assert all(
            r["speaker_party"] in speaker_party
            for r in DATASET.records(speaker_party=speaker_party, limit=10)
        )


def test_records_chamber():
    chambers = ({"House"}, {"House", "Senate"})
    for chamber in chambers:
        assert all(
            r["chamber"] in chamber for r in DATASET.records(chamber=chamber, limit=10)
        )


def test_records_congress():
    congresses = ({104}, {104, 114})
    for congress in congresses:
        assert all(
            r["congress"] in congress
            for r in DATASET.records(congress=congress, limit=10)
        )


def test_records_date_range():
    date_ranges = (["2000-01-01", "2001-01-01"], ("2010-01-01", "2010-02-01"))
    for date_range in date_ranges:
        assert all(
            date_range[0] <= r["date"] < date_range[1]
            for r in DATASET.records(date_range=date_range, limit=10)
        )


def test_bad_filters():
    bad_filters = (
        {"speaker_name": "Burton DeWilde"},
        {"speaker_party": "Whigs"},
        {"chamber": "Pot"},
        {"congress": 42},
        {"date_range": "2016-01-01"},
    )
    for bad_filter in bad_filters:
        with pytest.raises(ValueError):
            list(DATASET.texts(**bad_filter))
