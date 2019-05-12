from __future__ import absolute_import, unicode_literals

import datetime
import os

import pytest

from textacy import compat
from textacy.datasets.supreme_court import SupremeCourt

DATASET = SupremeCourt()

pytestmark = pytest.mark.skipif(
    DATASET.filepath is None,
    reason="SupremeCourt dataset must be downloaded before running tests",
)


@pytest.mark.skip("No need to download a new dataset every time")
def test_download(tmpdir):
    dataset = SupremeCourt(data_dir=str(tmpdir))
    dataset.download()
    assert os.path.isfile(dataset._filepath)


def test_oserror(tmpdir):
    dataset = SupremeCourt(data_dir=str(tmpdir))
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


def test_records_opinion_author():
    opinion_authors = ({78}, {78, 81})
    for opinion_author in opinion_authors:
        records = list(DATASET.records(opinion_author=opinion_author, limit=10))
        assert len(records) >= 1
        assert all(
            meta["maj_opinion_author"] in opinion_author
            for text, meta in records
        )


def test_records_decision_direction():
    decision_directions = ("liberal", {"conservative", "unspecifiable"})
    for decision_direction in decision_directions:
        records = list(DATASET.records(decision_direction=decision_direction, limit=10))
        assert len(records) >= 1
        assert all(
            meta["decision_direction"] in decision_direction
            for text, meta in records
        )


def test_records_issue_area():
    issue_areas = ({2}, {4, 5, 6})
    for issue_area in issue_areas:
        records = list(DATASET.records(issue_area=issue_area, limit=10))
        assert len(records) >= 1
        assert all(
            meta["issue_area"] in issue_area
            for text, meta in records
        )


def test_records_date_range():
    date_ranges = (["1947-01-01", "1948-01-01"], ("1947-07-01", "1947-12-31"))
    for date_range in date_ranges:
        records = list(DATASET.records(date_range=date_range, limit=10))
        assert len(records) >= 1
        assert all(
            date_range[0] <= meta["decision_date"] < date_range[1]
            for text, meta in records
        )


def test_bad_filters():
    bad_filters = (
        {"opinion_author": 1000},
        {"decision_direction": "blatantly political"},
        {"issue_area": 1000},
        {"min_len": -1},
    )
    for bad_filter in bad_filters:
        with pytest.raises(ValueError):
            list(DATASET.texts(**bad_filter))
    bad_filters = (
        {"opinion_author": "Burton DeWilde"},
        {"issue_area": "legalizing gay marriage, woo!"},
        {"date_range": "2016-01-01"},
        {"date_range": (datetime.date(2000, 1, 1), datetime.date(2001, 1, 1))},
    )
    for bad_filter in bad_filters:
        with pytest.raises(TypeError):
            list(DATASET.texts(**bad_filter))
