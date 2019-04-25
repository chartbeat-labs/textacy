from __future__ import absolute_import, unicode_literals

import datetime
import os

import pytest

from textacy import compat
from textacy.datasets.reddit_comments import RedditComments

DATASET = RedditComments()

pytestmark = pytest.mark.skipif(
    not DATASET.filepaths,
    reason="RedditComments dataset must be downloaded before running tests",
)


@pytest.mark.skip("No need to download a new dataset every time")
def test_download(tmpdir):
    dataset = RedditComments(data_dir=str(tmpdir))
    dataset.download()
    assert all(os.path.isfile(filepath) for filepath in dataset.filepaths)


def test_oserror(tmpdir):
    dataset = RedditComments(data_dir=str(tmpdir))
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


def test_records_subreddit():
    subreddits = ({"politics"}, {"politics", "programming"})
    for subreddit in subreddits:
        records = list(DATASET.records(subreddit=subreddit, limit=10))
        assert all(
            meta["subreddit"] in subreddit
            for text, meta in records
        )


def test_records_date_range():
    date_ranges = (("2007-10", "2007-12"), ["2007-10-01", "2008-01-01"])
    for date_range in date_ranges:
        records = list(DATASET.records(date_range=date_range, limit=10))
        assert all(
            date_range[0] <= meta["created_utc"] < date_range[1]
            for text, meta in records
        )


def test_records_score_range():
    score_ranges = ([-10, 10], (5, 100))
    for score_range in score_ranges:
        records = list(DATASET.records(score_range=score_range, limit=10))
        assert all(
            score_range[0] <= meta["score"] < score_range[1]
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
        {"date_range": "2016-01-01"},
        {"date_range": (datetime.date(2007, 10, 1), datetime.date(2007, 11, 1))},
        {"score_range": 10},
        {"score_range": ["low", "high"]},
    )
    for bad_filter in bad_filters:
        with pytest.raises(TypeError):
            list(DATASET.texts(**bad_filter))
