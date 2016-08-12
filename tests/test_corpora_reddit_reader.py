from __future__ import absolute_import, unicode_literals

import os
import shutil
import tempfile
import unittest

from textacy.compat import PY2, unicode_type
from textacy.corpora import RedditReader
from textacy.fileio import write_json_lines


REDDIT_COMMENTS = [
    {"author_flair_css_class": None, "created_utc": "1420070400", "controversiality": 0, "parent_id": "t3_2qyr1a", "score": 14, "author": "YoungModern", "subreddit_id": "t5_2r0gj", "gilded": 0, "distinguished": None, "id": "cnas8zv", "link_id": "t3_2qyr1a", "name": "t1_cnas8zv", "author_flair_text": None, "downs": 0, "subreddit": "exmormon", "ups": 14, "edited": False, "retrieved_on": 1425124282, "body": "Most of us have some family members like this. *Most* of my family is like this. ", "score_hidden": False, "archived": False},
    {"author_flair_css_class": "on", "created_utc": "1420070400", "parent_id": "t1_cnas2b6", "downs": 0, "score": 3, "author_flair_text": "Ontario", "subreddit_id": "t5_2s4gt", "gilded": 0, "distinguished": None, "id": "cnas8zw", "link_id": "t3_2qv6c6", "author": "RedCoatsForever", "controversiality": 0, "retrieved_on": 1425124282, "archived": False, "subreddit": "CanadaPolitics", "edited": False, "ups": 3, "body": "But Mill's career was way better. Bentham is like, the Joseph Smith to Mill's Brigham Young.", "score_hidden": False, "name": "t1_cnas8zw"},
    {"author_flair_css_class": None, "created_utc": "1420070400", "parent_id": "t3_2qxefp", "author_flair_text": None, "score": 1, "subreddit_id": "t5_2s7tt", "gilded": 0, "distinguished": None, "id": "cnas8zx", "link_id": "t3_2qxefp", "author": "vhisic", "name": "t1_cnas8zx", "retrieved_on": 1425124282, "downs": 0, "subreddit": "AdviceAnimals", "controversiality": 0, "edited": False, "ups": 1, "body": "Mine uses a strait razor, and as much as i love the clippers i love the razor so much more. Then he follows it up with a warm towel. \nI think i might go get a hair cut this week.", "score_hidden": False, "archived": False},
    ]


class RedditReaderTestCase(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp(
            prefix='test_corpora', dir=os.path.dirname(os.path.abspath(__file__)))
        reddit_fname = os.path.join(self.tempdir, 'RC_test.bz2')
        if PY2 is False:
            write_json_lines(REDDIT_COMMENTS, reddit_fname, mode='wt',
                             auto_make_dirs=True)
        else:
            write_json_lines(REDDIT_COMMENTS, reddit_fname, mode='wb',
                             auto_make_dirs=True)
        self.redditreader = RedditReader(reddit_fname)

    def test_texts(self):
        for text in self.redditreader.texts():
            self.assertIsInstance(text, unicode_type)

    def test_texts_limit(self):
        texts = list(self.redditreader.texts(limit=1))
        self.assertEqual(len(texts), 1)

    def test_texts_min_len(self):
        for text in self.redditreader.texts(min_len=100):
            self.assertTrue(len(text) >= 100)

    def test_records(self):
        for record in self.redditreader.records():
            self.assertIsInstance(record, dict)

    def test_records_limit(self):
        records = list(self.redditreader.records(limit=1))
        self.assertEqual(len(records), 1)

    def test_records_score_range(self):
        score_ranges = [(-2, 2), (5, None), (None, 2)]
        for score_range in score_ranges:
            records = list(self.redditreader.records(score_range=score_range))
            self.assertEqual(len(records), 1)
            for record in records:
                if score_range[0]:
                    self.assertTrue(record['score'] >= score_range[0])
                if score_range[1]:
                    self.assertTrue(record['score'] <= score_range[1])

    def test_records_subreddit(self):
        subreddits = [('exmormon',), {'CanadaPolitics', 'AdviceAnimals'}]
        expected_lens = (1, 2)
        for subreddit, expected_len in zip(subreddits, expected_lens):
            records = list(self.redditreader.records(subreddit=subreddit))
            self.assertEqual(len(records), expected_len)
            for record in records:
                self.assertTrue(record['subreddit'] in subreddit)

    def tearDown(self):
        shutil.rmtree(self.tempdir)
