from __future__ import absolute_import, unicode_literals

import os
import shutil
import tempfile
import unittest

from textacy import data_dir
from textacy.compat import unicode_
from textacy.datasets.capitol_words import CapitolWords

DATASET = CapitolWords()


@unittest.skipUnless(
    DATASET.filename, 'CapitolWords dataset must be downloaded before running tests')
class CapitolWordsTestCase(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp(
            prefix='test_datasets_', dir=os.path.dirname(os.path.abspath(__file__)))

    @unittest.skip("No need to download a new dataset every time")
    def test_download(self):
        dataset = CapitolWords(data_dir=self.tempdir)
        dataset.download()
        self.assertTrue(os.path.exists(dataset.filename))

    def test_ioerror(self):
        dataset = CapitolWords(data_dir=self.tempdir)
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

    def test_records_speaker_name(self):
        speaker_names = ({'Bernie Sanders'}, {'Ted Cruz', 'Barack Obama'})
        for speaker_name in speaker_names:
            self.assertTrue(
                all(r['speaker_name'] in speaker_name
                    for r in DATASET.records(speaker_name=speaker_name, limit=10)))

    def test_records_speaker_party(self):
        speaker_parties = ({'R'}, {'D', 'I'})
        for speaker_party in speaker_parties:
            self.assertTrue(
                all(r['speaker_party'] in speaker_party
                    for r in DATASET.records(speaker_party=speaker_party, limit=10)))

    def test_records_chamber(self):
        chambers = ({'House'}, {'House', 'Senate'})
        for chamber in chambers:
            self.assertTrue(
                all(r['chamber'] in chamber
                    for r in DATASET.records(chamber=chamber, limit=10)))

    def test_records_congress(self):
        congresses = ({104}, {104, 114})
        for congress in congresses:
            self.assertTrue(
                all(r['congress'] in congress
                    for r in DATASET.records(congress=congress, limit=10)))

    def test_records_date_range(self):
        date_ranges = (
            ['2000-01-01', '2001-01-01'],
            ('2010-01-01', '2010-02-01'),
            )
        for date_range in date_ranges:
            self.assertTrue(
                all(date_range[0] <= r['date'] < date_range[1]
                    for r in DATASET.records(date_range=date_range, limit=10)))

    def test_bad_filters(self):
        bad_filters = ({'speaker_name': 'Burton DeWilde'},
                       {'speaker_party': 'Whigs'},
                       {'chamber': 'Pot'},
                       {'congress': 42},
                       {'date_range': '2016-01-01'})
        for bad_filter in bad_filters:
            with self.assertRaises(ValueError):
                list(DATASET.texts(**bad_filter))

    def tearDown(self):
        shutil.rmtree(self.tempdir)
