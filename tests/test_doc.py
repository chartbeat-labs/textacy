# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

import unittest

from textacy import Doc
from textacy import data


class DocTestCase(unittest.TestCase):

    def test_unicode_content(self):
        self.assertIsInstance(Doc('This is an English sentence.'), Doc)

    def test_spacydoc_content(self):
        spacy_lang = data.load_spacy('en_core_web_sm')
        spacy_doc = spacy_lang('This is an English sentence.')
        self.assertIsInstance(Doc(spacy_doc), Doc)

    def test_invalid_content(self):
        with self.assertRaises(ValueError):
            Doc(b'This is an English sentence in bytes.')
            Doc({'content': 'This is an English sentence as dict value.'})
            Doc(True)

    def test_lang_str(self):
        self.assertIsInstance(
            Doc('This is an English sentence.', lang='en'), Doc)

    def test_lang_spacylang(self):
        spacy_lang = data.load_spacy('en_core_web_sm')
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
        with self.assertRaises(ValueError):
            Doc('This is an English sentence.', lang=b'en')
            Doc('This is an English sentence.', lang=['en', 'en_core_web_sm'])
            Doc('This is an English sentence.', lang=True)

    def test_invalid_content_lang_combo(self):
        spacy_lang = data.load_spacy('en_core_web_sm')
        with self.assertRaises(ValueError):
            Doc(spacy_lang('Hola, cómo estás mi amigo?'), lang='es')
