from __future__ import absolute_import, unicode_literals

import unittest

import textacy


class DataTestCase(unittest.TestCase):

    def test_load_pyphen(self):
        for lang in ('en', 'de'):
            _ = textacy.data.load_hyphenator(lang=lang)

    def test_load_depechemood(self):
        for weighting in ('freq', 'normfreq', 'tfidf'):
            _ = textacy.data.load_depechemood(weighting=weighting)
