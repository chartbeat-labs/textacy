from __future__ import absolute_import, unicode_literals

import os
import shutil
import tempfile
import unittest

from textacy import data_dir
from textacy.compat import unicode_
from textacy.datasets.oxford_text_archive import OxfordTextArchive

DATASET = OxfordTextArchive()


@unittest.skipUnless(
    DATASET.filename, 'OxfordTextArchive dataset must be downloaded before running tests')
class OxfordTextArchiveTestCase(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp(
            prefix='test_datasets_', dir=os.path.dirname(os.path.abspath(__file__)))

    @unittest.skip("No need to download a new dataset every time")
    def test_download(self):
        dataset = OxfordTextArchive(data_dir=self.tempdir)
        dataset.download()
        self.assertTrue(os.path.exists(dataset.filename))

    def test_ioerror(self):
        dataset = OxfordTextArchive(data_dir=self.tempdir)
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

    def test_records_author(self):
        authors = ({'Shakespeare, William'}, {'Wollstonecraft, Mary', 'Twain, Mark'})
        for author in authors:
            self.assertTrue(
                all(a in author
                    for r in DATASET.records(author=author, limit=10)
                    for a in r['author']))

    def test_records_date_range(self):
        date_ranges = (
            ['1900-01-01', '1950-01-01'],
            ('1600-01-01', '1700-01-01'),
            )
        for date_range in date_ranges:
            self.assertTrue(
                all(date_range[0] <= r['year'] < date_range[1]
                    for r in DATASET.records(date_range=date_range, limit=10)))

    def test_bad_filters(self):
        bad_filters = ({'author': 'Burton DeWilde'},
                       {'date_range': '2016-01-01'})
        for bad_filter in bad_filters:
            with self.assertRaises(ValueError):
                list(DATASET.texts(**bad_filter))

    def tearDown(self):
        shutil.rmtree(self.tempdir)
