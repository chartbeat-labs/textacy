from __future__ import absolute_import, unicode_literals

import os
import shutil
import tempfile
import unittest

from textacy.compat import unicode_type
from textacy.corpora import supremecourt


class SupremeCourtTestCase(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp(
            prefix='test_corpora', dir=os.path.dirname(os.path.abspath(__file__)))

    def test_supremecourt_oserror(self):
        self.assertRaises(
            OSError, supremecourt.SupremeCourt,
            self.tempdir, False)

    @unittest.skip("no need to download a fresh corpus every time")
    def test_supremecourt_download(self):
        supremecourt.SupremeCourt(
            data_dir=self.tempdir, download_if_missing=True)
        self.assertTrue(
            os.path.exists(os.path.join(self.tempdir, 'supremecourt', supremecourt.FILENAME)))

    def test_supremecourt_texts(self):
        cw = supremecourt.SupremeCourt(download_if_missing=True)
        for text in cw.texts(limit=3):
            self.assertIsInstance(text, unicode_type)

    def test_supremecourt_texts_limit(self):
        cw = supremecourt.SupremeCourt(download_if_missing=True)
        for limit in (1, 5, 100):
            self.assertEqual(sum(1 for _ in cw.texts(limit=limit)), limit)

    def test_supremecourt_texts_min_len(self):
        cw = supremecourt.SupremeCourt(download_if_missing=True)
        for min_len in (100, 200, 1000):
            self.assertTrue(
                all(len(text) >= min_len
                    for text in cw.texts(min_len=min_len, limit=1000)))

    def test_supremecourt_docs(self):
        cw = supremecourt.SupremeCourt(download_if_missing=True)
        for doc in cw.docs(limit=3):
            self.assertIsInstance(doc, dict)

    def test_supremecourt_docs_opinion_author(self):
        cw = supremecourt.SupremeCourt(download_if_missing=True)
        opinion_authors = ({109}, {113, 114})
        for opinion_author in opinion_authors:
            self.assertTrue(
                all(d['maj_opinion_author'] in opinion_author
                    for d in cw.docs(opinion_author=opinion_author, limit=100)))

    def test_supremecourt_docs_decision_direction(self):
        cw = supremecourt.SupremeCourt(download_if_missing=True)
        decision_directions = ('liberal', {'conservative', 'unspecifiable'})
        for decision_direction in decision_directions:
            self.assertTrue(
                all(d['decision_direction'] in decision_direction
                    for d in cw.docs(decision_direction=decision_direction, limit=100)))

    def test_supremecourt_docs_issue_area(self):
        cw = supremecourt.SupremeCourt(download_if_missing=True)
        issue_areas = ({2}, {4, 5, 6})
        for issue_area in issue_areas:
            self.assertTrue(
                all(d['issue_area'] in issue_area
                    for d in cw.docs(issue_area=issue_area, limit=100)))

    def test_supremecourt_bad_filters(self):
        cw = supremecourt.SupremeCourt(download_if_missing=True)
        bad_filters = ({'opinion_author': 'Burton DeWilde'},
                       {'opinion_author': 1000},
                       {'decision_direction': 'blatantly political'},
                       {'issue_area': 'legalizing gay marriage, woo!'},
                       {'issue_area': 1000},
                       {'date_range': '2016-01-01'})
        for bad_filter in bad_filters:
            with self.assertRaises(ValueError):
                list(cw._iterate(True, **bad_filter))

    def tearDown(self):
        shutil.rmtree(self.tempdir)
