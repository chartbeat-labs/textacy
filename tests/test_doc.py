# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

import unittest

from textacy import Doc
from textacy import compat
from textacy import data

if compat.is_python2:
    int_types = (int, long)
else:
    int_types = int

TEXT = """
Since the so-called "statistical revolution" in the late 1980s and mid 1990s, much Natural Language Processing research has relied heavily on machine learning.
Formerly, many language-processing tasks typically involved the direct hand coding of rules, which is not in general robust to natural language variation. The machine-learning paradigm calls instead for using statistical inference to automatically learn such rules through the analysis of large corpora of typical real-world examples (a corpus is a set of documents, possibly with human or computer annotations).
Many different classes of machine learning algorithms have been applied to NLP tasks. These algorithms take as input a large set of "features" that are generated from the input data. Some of the earliest-used algorithms, such as decision trees, produced systems of hard if-then rules similar to the systems of hand-written rules that were then common. Increasingly, however, research has focused on statistical models, which make soft, probabilistic decisions based on attaching real-valued weights to each input feature. Such models have the advantage that they can express the relative certainty of many different possible answers rather than only one, producing more reliable results when such a model is included as a component of a larger system.
"""


class DocInitTestCase(unittest.TestCase):

    def test_unicode_content(self):
        self.assertIsInstance(Doc('This is an English sentence.'), Doc)

    def test_spacydoc_content(self):
        spacy_lang = data.load_spacy('en')
        spacy_doc = spacy_lang('This is an English sentence.')
        self.assertIsInstance(Doc(spacy_doc), Doc)

    def test_invalid_content(self):
        invalid_contents = [
            b'This is an English sentence in bytes.',
            {'content': 'This is an English sentence as dict value.'},
            True,
            ]
        for invalid_content in invalid_contents:
            with self.assertRaises(ValueError):
                Doc(invalid_content)

    def test_lang_str(self):
        self.assertIsInstance(
            Doc('This is an English sentence.', lang='en'), Doc)

    def test_lang_spacylang(self):
        spacy_lang = data.load_spacy('en')
        self.assertIsInstance(
            Doc('This is an English sentence.', lang=spacy_lang), Doc)

    def test_lang_callable(self):
        def dumb_detect_language(text):
            return 'en'
        self.assertIsInstance(
            Doc('This is an English sentence.', lang=dumb_detect_language), Doc)
        self.assertIsInstance(
            Doc('This is an English sentence.', lang=lambda x: 'en'), Doc)

    def test_invalid_lang(self):
        invalid_langs = [b'en', ['en', 'en_core_web_sm'], True]
        for invalid_lang in invalid_langs:
            with self.assertRaises(ValueError):
                Doc('This is an English sentence.', lang=invalid_lang)

    def test_invalid_content_lang_combo(self):
        spacy_lang = data.load_spacy('en')
        with self.assertRaises(ValueError):
            Doc(spacy_lang('Hola, cómo estás mi amigo?'), lang='es')


class DocMethodsTestCase(unittest.TestCase):

    def setUp(self):
        self.doc = Doc(TEXT.strip(), lang='en')

    def test_n_tokens_and_sents(self):
        self.assertEqual(self.doc.n_tokens, 241)
        self.assertEqual(self.doc.n_sents, 8)

    def test_term_count(self):
        self.assertEqual(self.doc.count('statistical'), 3)
        self.assertEqual(self.doc.count('machine learning'), 2)
        self.assertEqual(self.doc.count('foo'), 0)

    def test_tokenized_text(self):
        tokenized_text = self.doc.tokenized_text
        self.assertIsInstance(tokenized_text, list)
        self.assertIsInstance(tokenized_text[0], list)
        self.assertIsInstance(tokenized_text[0][0], compat.unicode_)
        self.assertEqual(len(tokenized_text), self.doc.n_sents)

    def test_pos_tagged_text(self):
        pos_tagged_text = self.doc.pos_tagged_text
        self.assertIsInstance(pos_tagged_text, list)
        self.assertIsInstance(pos_tagged_text[0], list)
        self.assertIsInstance(pos_tagged_text[0][0], tuple)
        self.assertIsInstance(pos_tagged_text[0][0][0], compat.unicode_)
        self.assertEqual(len(pos_tagged_text), self.doc.n_sents)

    def test_to_terms_list(self):
        full_terms_list = list(self.doc.to_terms_list(as_strings=True))
        full_terms_list_ids = list(self.doc.to_terms_list(as_strings=False))
        self.assertEqual(len(full_terms_list), len(full_terms_list_ids))
        self.assertIsInstance(full_terms_list[0], compat.unicode_)
        self.assertIsInstance(full_terms_list_ids[0], int)
        self.assertNotEqual(
            full_terms_list[0],
            list(self.doc.to_terms_list(as_strings=True, normalize=False))[0])
        self.assertLess(
            len(list(self.doc.to_terms_list(ngrams=False))),
            len(full_terms_list))
        self.assertLess(
            len(list(self.doc.to_terms_list(ngrams=1))),
            len(full_terms_list))
        self.assertLess(
            len(list(self.doc.to_terms_list(ngrams=(1, 2)))),
            len(full_terms_list))
        self.assertLess(
            len(list(self.doc.to_terms_list(ngrams=False))),
            len(full_terms_list))

    def test_to_bag_of_words(self):
        bow = self.doc.to_bag_of_words(weighting='count')
        self.assertIsInstance(bow, dict)
        self.assertIsInstance(list(bow.keys())[0], int_types)
        self.assertIsInstance(list(bow.values())[0], int)
        bow = self.doc.to_bag_of_words(weighting='binary')
        self.assertIsInstance(bow, dict)
        self.assertIsInstance(list(bow.keys())[0], int_types)
        self.assertIsInstance(list(bow.values())[0], int)
        for value in list(bow.values())[0:10]:
            self.assertLess(value, 2)
        bow = self.doc.to_bag_of_words(weighting='freq')
        self.assertIsInstance(bow, dict)
        self.assertIsInstance(list(bow.keys())[0], int_types)
        self.assertIsInstance(list(bow.values())[0], float)
        bow = self.doc.to_bag_of_words(as_strings=True)
        self.assertIsInstance(bow, dict)
        self.assertIsInstance(list(bow.keys())[0], compat.unicode_)
