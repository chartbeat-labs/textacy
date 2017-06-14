from __future__ import absolute_import, unicode_literals

import os
import shutil
import tempfile
import unittest

from textacy import data_dir
from textacy.compat import unicode_
from textacy.datasets.supreme_court import SupremeCourt

DATASET = SupremeCourt()


@unittest.skipUnless(
    DATASET.filename, 'SupremeCourt dataset must be downloaded before running tests')
class SupremeCourtTestCase(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp(
            prefix='test_datasets_', dir=os.path.dirname(os.path.abspath(__file__)))

    @unittest.skip("No need to download a new dataset every time")
    def test_download(self):
        dataset = SupremeCourt(data_dir=self.tempdir)
        dataset.download()
        self.assertTrue(os.path.exists(dataset.filename))

    def test_ioerror(self):
        dataset = SupremeCourt(data_dir=self.tempdir)
        with self.assertRaises(IOError):
            _ = list(dataset.texts())

    def test_texts(self):
        for text in DATASET.texts(limit=3):
            self.assertIsInstance(text, unicode_)

    def test_texts_limit(self):
        for limit in (1, 5, 10):
            self.assertEqual(sum(1 for _ in DATASET.texts(limit=limit)), limit)

    def test_texts_min_len(self):
        for min_len in (100, 200, 1000):
            self.assertTrue(
                all(len(text) >= min_len
                    for text in DATASET.texts(min_len=min_len, limit=10)))

    def test_records(self):
        for record in DATASET.records(limit=3):
            self.assertIsInstance(record, dict)

    def test_records_opinion_author(self):
        opinion_authors = ({109}, {113, 114})
        for opinion_author in opinion_authors:
            self.assertTrue(
                all(r['maj_opinion_author'] in opinion_author
                    for r in DATASET.records(opinion_author=opinion_author, limit=10)))

    def test_records_decision_direction(self):
        decision_directions = ('liberal', {'conservative', 'unspecifiable'})
        for decision_direction in decision_directions:
            self.assertTrue(
                all(r['decision_direction'] in decision_direction
                    for r in DATASET.records(decision_direction=decision_direction, limit=10)))

    def test_records_issue_area(self):
        issue_areas = ({2}, {4, 5, 6})
        for issue_area in issue_areas:
            self.assertTrue(
                all(r['issue_area'] in issue_area
                    for r in DATASET.records(issue_area=issue_area, limit=10)))

    def test_records_date_range(self):
        date_ranges = (
            ['1970-01-01', '1980-01-01'],
            ('2000-01-01', '2000-02-01'),
            )
        for date_range in date_ranges:
            self.assertTrue(
                all(date_range[0] <= r['decision_date'] < date_range[1]
                    for r in DATASET.records(date_range=date_range, limit=10)))

    def test_bad_filters(self):
        bad_filters = ({'opinion_author': 'Burton DeWilde'},
                       {'opinion_author': 1000},
                       {'decision_direction': 'blatantly political'},
                       {'issue_area': 'legalizing gay marriage, woo!'},
                       {'issue_area': 1000},
                       {'date_range': '2016-01-01'})
        for bad_filter in bad_filters:
            with self.assertRaises(ValueError):
                list(DATASET.texts(**bad_filter))

    def tearDown(self):
        shutil.rmtree(self.tempdir)
