from __future__ import absolute_import, unicode_literals

import os
import tempfile
import unittest

from textacy.corpora import bernie_and_hillary


class CorporaTestCase(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp(
            prefix='test_corpora', dir=os.path.dirname(os.path.abspath(__file__)))

    def test_fetch_bernie_and_hillary_exception(self):
        self.assertRaises(
            IOError, bernie_and_hillary.fetch_bernie_and_hillary,
            self.tempdir, False)

    def test_download_bernie_and_hillary(self):
        bernie_and_hillary._download_bernie_and_hillary(data_dir=self.tempdir)
        self.assertTrue(
            os.path.exists(os.path.join(self.tempdir, bernie_and_hillary.FNAME)))

    def test_fetch_bernie_and_hillary(self):
        bnh = bernie_and_hillary.fetch_bernie_and_hillary(data_dir=self.tempdir)
        self.assertIsInstance(bnh, list)
        self.assertEqual(len(bnh), 3066)

    def test_fetch_bernie_and_hillary_shuffle(self):
        bnh = bernie_and_hillary.fetch_bernie_and_hillary(
            data_dir=self.tempdir, shuffle=True)
        # technically, this test has a failure probability of 1/3066
        self.assertNotEqual(bnh[0]['date'], '1996-01-04')

    def tearDown(self):
        for fname in os.listdir(self.tempdir):
            os.remove(os.path.join(self.tempdir, fname))
        os.rmdir(self.tempdir)
