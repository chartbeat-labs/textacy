from __future__ import absolute_import, unicode_literals

import os
import shutil
import tempfile
import unittest

from textacy import data_dir
from textacy.compat import unicode_
from textacy.datasets.reddit_comments import RedditComments

DATASET = RedditComments()


@unittest.skipUnless(
    DATASET.filenames, 'RedditComments dataset must be downloaded before running tests')
class RedditCommentsTestCase(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp(
            prefix='test_datasets_', dir=os.path.dirname(os.path.abspath(__file__)))

    @unittest.skip("No need to download a new dataset every time")
    def test_download(self):
        dataset = RedditComments(data_dir=self.tempdir)
        dataset.download()
        self.assertTrue(
            all(os.path.exists(filename)
                for filename in self.filenames))

    def test_ioerror(self):
        dataset = RedditComments(data_dir=self.tempdir)
        with self.assertRaises(IOError):
            _ = list(dataset.texts())

    def test_texts(self):
        for text in DATASET.texts(limit=3):
            self.assertIsInstance(text, unicode_)

    def test_texts_limit(self):
        for limit in (1, 5, 100):
            self.assertEqual(sum(1 for _ in DATASET.texts(limit=limit)), limit)

    def test_texts_min_len(self):
        for min_len in (100, 200, 1000):
            self.assertTrue(
                all(len(text) >= min_len
                    for text in DATASET.texts(min_len=min_len, limit=10)))

    def test_records(self):
        for record in DATASET.records(limit=3):
            self.assertIsInstance(record, dict)

    def test_records_subreddit(self):
        subreddits = ({'exmormon'}, {'CanadaPolitics', 'AdviceAnimals'})
        for subreddit in subreddits:
            self.assertTrue(
                all(r['subreddit'] in subreddit
                    for r in DATASET.records(subreddit=subreddit, limit=10)))

    def test_records_date_range(self):
        date_ranges = (
            ['2007-10-01', '2008-01-01'],
            ('2007-10-01', '2007-11-01'),
            )
        for date_range in date_ranges:
            self.assertTrue(
                all(date_range[0] <= r['created_utc'] < date_range[1]
                    for r in DATASET.records(date_range=date_range, limit=10)))

    def test_records_score_range(self):
        score_ranges = (
            [-10, 10],
            (5, 100),
            )
        for score_range in score_ranges:
            self.assertTrue(
                all(score_range[0] <= r['score'] < score_range[1]
                    for r in DATASET.records(score_range=score_range, limit=10)))

    def test_bad_filters(self):
        bad_filters = ({'date_range': '2016-01-01'},
                       {'score_range': 10})
        for bad_filter in bad_filters:
            with self.assertRaises(ValueError):
                list(DATASET.texts(**bad_filter))

    def tearDown(self):
        shutil.rmtree(self.tempdir)
