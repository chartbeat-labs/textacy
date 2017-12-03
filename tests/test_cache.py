from __future__ import absolute_import, unicode_literals

import unittest

from textacy import cache


class CacheTestCase(unittest.TestCase):

    def test_load_spacy(self):
        for lang in ('en', 'en_core_web_sm'):
            for disable in (None, ('parser', 'ner')):
                _ = cache.load_spacy(lang, disable=disable)

    def test_load_spacy_hashability(self):
        with self.assertRaises(TypeError):
            _ = cache.load_spacy('en', disable=['tagger', 'parser', 'ner'])

    def test_load_pyphen(self):
        for lang in ('en', 'es'):
            _ = cache.load_hyphenator(lang=lang)

    def test_load_depechemood(self):
        for weighting in ('freq', 'normfreq', 'tfidf'):
            _ = cache.load_depechemood(weighting=weighting)

    def test_cache(self):
        _ = cache.load_hyphenator(lang='en')
        _ = cache.load_spacy('en')
        self.assertTrue(len(cache.LRU_CACHE.keys()) >= 2)
        # check cache size; low thresh but still larger than if the size of
        # loaded data was not being correctly assessed
        self.assertTrue(cache.LRU_CACHE.currsize >= 1000)
