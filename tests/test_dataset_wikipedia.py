from __future__ import absolute_import, unicode_literals

import os
import shutil
import tempfile
import unittest

from textacy import data_dir
from textacy.compat import unicode_
from textacy.datasets.wikipedia import Wikipedia

DATASET = Wikipedia(lang='en', version='latest')


@unittest.skipUnless(
    DATASET.filename, 'Wikipedia dataset must be downloaded before running tests')
class WikipediaTestCase(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp(
            prefix='test_datasets_', dir=os.path.dirname(os.path.abspath(__file__)))

    @unittest.skip("No need to download a new dataset every time")
    def test_download(self):
        dataset = Wikipedia(data_dir=self.tempdir)
        dataset.download()
        self.assertTrue(os.path.exists(dataset.filename))

    def test_ioerror(self):
        dataset = Wikipedia(data_dir=self.tempdir)
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

    def test_records_fast(self):
        for record in DATASET.records(limit=3, fast=True):
            self.assertIsInstance(record, dict)

    # TODO: test individual parsing functions

    def tearDown(self):
        shutil.rmtree(self.tempdir)
