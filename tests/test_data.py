from __future__ import absolute_import, unicode_literals

import unittest

import textacy


class DataTestCase(unittest.TestCase):

    def test_load_spacy(self):
        for lang in ('en', 'en_core_web_sm'):
            for disable in (None, ('parser', 'ner')):
                _ = textacy.data.load_spacy(lang, disable=disable)

    def test_load_spacy_hashability(self):
        with self.assertRaises(TypeError):
            _ = textacy.data.load_spacy('en', disable=['tagger', 'parser', 'ner'])

    def test_load_pyphen(self):
        for lang in ('en', 'de', 'es'):
            _ = textacy.data.load_hyphenator(lang=lang)

    def test_load_depechemood(self):
        for weighting in ('freq', 'normfreq', 'tfidf'):
            _ = textacy.data.load_depechemood(weighting=weighting)
