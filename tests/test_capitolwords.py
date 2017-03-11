from __future__ import absolute_import, unicode_literals

import os
import shutil
import tempfile
import unittest

from textacy import __resources_dir__
from textacy.compat import unicode_
from textacy.corpora import capitolwords

CORPUS_FILEPATH = os.path.join(__resources_dir__, 'capitolwords', capitolwords.FILENAME)


@unittest.skipUnless(
    os.path.isfile(CORPUS_FILEPATH), 'CapitolWords corpus must first be downloaded to run tests')
class CapitolWordsTestCase(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp(
            prefix='test_corpora', dir=os.path.dirname(os.path.abspath(__file__)))

    def test_capitolwords_oserror(self):
        self.assertRaises(
            OSError, capitolwords.CapitolWords,
            self.tempdir, False)

    @unittest.skip("no need to download a fresh corpus every time")
    def test_capitolwords_download(self):
        capitolwords.CapitolWords(
            data_dir=self.tempdir, download_if_missing=False)
        self.assertTrue(
            os.path.exists(os.path.join(self.tempdir, 'capitolwords', capitolwords.FILENAME)))

    def test_capitolwords_texts(self):
        cw = capitolwords.CapitolWords(download_if_missing=False)
        for text in cw.texts(limit=3):
            self.assertIsInstance(text, unicode_)

    def test_capitolwords_texts_limit(self):
        cw = capitolwords.CapitolWords(download_if_missing=False)
        for limit in (1, 5, 100):
            self.assertEqual(sum(1 for _ in cw.texts(limit=limit)), limit)

    def test_capitolwords_texts_min_len(self):
        cw = capitolwords.CapitolWords(download_if_missing=False)
        for min_len in (100, 200, 1000):
            self.assertTrue(
                all(len(text) >= min_len
                    for text in cw.texts(min_len=min_len, limit=1000)))

    def test_capitolwords_records(self):
        cw = capitolwords.CapitolWords(download_if_missing=False)
        for record in cw.records(limit=3):
            self.assertIsInstance(record, dict)

    def test_capitolwords_records_speaker_name(self):
        cw = capitolwords.CapitolWords(download_if_missing=False)
        speaker_names = ({'Bernie Sanders'}, {'Ted Cruz', 'Barack Obama'})
        for speaker_name in speaker_names:
            self.assertTrue(
                all(r['speaker_name'] in speaker_name
                    for r in cw.records(speaker_name=speaker_name, limit=1000)))

    def test_capitolwords_records_speaker_party(self):
        cw = capitolwords.CapitolWords(download_if_missing=False)
        speaker_parties = ({'R'}, {'D', 'I'})
        for speaker_party in speaker_parties:
            self.assertTrue(
                all(r['speaker_party'] in speaker_party
                    for r in cw.records(speaker_party=speaker_party, limit=1000)))

    def test_capitolwords_records_chamber(self):
        cw = capitolwords.CapitolWords(download_if_missing=False)
        chambers = ({'House'}, {'House', 'Senate'})
        for chamber in chambers:
            self.assertTrue(
                all(r['chamber'] in chamber
                    for r in cw.records(chamber=chamber, limit=1000)))

    def test_capitolwords_records_congress(self):
        cw = capitolwords.CapitolWords(download_if_missing=False)
        congresses = ({104}, {104, 114})
        for congress in congresses:
            self.assertTrue(
                all(r['congress'] in congress
                    for r in cw.records(congress=congress, limit=1000)))

    def test_capitolwords_bad_filters(self):
        cw = capitolwords.CapitolWords(download_if_missing=False)
        bad_filters = ({'speaker_name': 'Burton DeWilde'},
                       {'speaker_party': 'Whigs'},
                       {'chamber': 'Pot'},
                       {'congress': 42},
                       {'date_range': '2016-01-01'})
        for bad_filter in bad_filters:
            with self.assertRaises(ValueError):
                list(cw._iterate(True, **bad_filter))

    def tearDown(self):
        shutil.rmtree(self.tempdir)
