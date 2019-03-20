from __future__ import absolute_import, unicode_literals

import os

import pytest

from textacy import compat
from textacy.datasets.supreme_court import SupremeCourt

DATASET = SupremeCourt()

pytestmark = pytest.mark.skipif(
    DATASET.filename is None,
    reason="SupremeCourt dataset must be downloaded before running tests",
)


@pytest.mark.skip("No need to download a new dataset every time")
def test_download(tmpdir):
    dataset = SupremeCourt(data_dir=str(tmpdir))
    dataset.download()
    assert os.path.exists(dataset.filename)


def test_ioerror(tmpdir):
    dataset = SupremeCourt(data_dir=str(tmpdir))
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


def test_records_opinion_author():
    opinion_authors = ({78}, {78, 81})
    for opinion_author in opinion_authors:
        assert all(
            r["maj_opinion_author"] in opinion_author
            for r in DATASET.records(opinion_author=opinion_author, limit=10)
        )


def test_records_decision_direction():
    decision_directions = ("liberal", {"conservative", "unspecifiable"})
    for decision_direction in decision_directions:
        assert all(
            r["decision_direction"] in decision_direction
            for r in DATASET.records(decision_direction=decision_direction, limit=10)
        )


def test_records_issue_area():
    issue_areas = ({2}, {4, 5, 6})
    for issue_area in issue_areas:
        assert all(
            r["issue_area"] in issue_area
            for r in DATASET.records(issue_area=issue_area, limit=10)
        )


def test_records_date_range():
    date_ranges = (["1970-01-01", "1971-01-01"], ("1971-07-01", "1971-12-31"))
    for date_range in date_ranges:
        assert all(
            date_range[0] <= r["decision_date"] < date_range[1]
            for r in DATASET.records(date_range=date_range, limit=10)
        )


def test_bad_filters():
    bad_filters = (
        {"opinion_author": "Burton DeWilde"},
        {"opinion_author": 1000},
        {"decision_direction": "blatantly political"},
        {"issue_area": "legalizing gay marriage, woo!"},
        {"issue_area": 1000},
        {"date_range": "2016-01-01"},
    )
    for bad_filter in bad_filters:
        with pytest.raises(ValueError):
            list(DATASET.texts(**bad_filter))
