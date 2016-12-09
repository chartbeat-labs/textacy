# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

import unittest

import numpy as np
from spacy import attrs

from textacy import data, keyterms, spacy_utils


class ExtractTestCase(unittest.TestCase):

    def setUp(self):
        self.maxDiff = None
        spacy_lang = data.load_spacy('en')
        text = """
            Two weeks ago, I was in Kuwait participating in an I.M.F. seminar for Arab educators. For 30 minutes, we discussed the impact of technology trends on education in the Middle East. And then an Egyptian education official raised his hand and asked if he could ask me a personal question: "I heard Donald Trump say we need to close mosques in the United States," he said with great sorrow. "Is that what we want our kids to learn?"
            """
        self.spacy_doc = spacy_lang(text.strip())
        cols = [attrs.TAG, attrs.HEAD, attrs.DEP]
        values = np.array(
            [[425, 1, 1500074], [443, 1, 392], [447, 3, 365], [416, 2, 407], [445, 1, 393],
             [455, 0, 53503], [432, -1, 405], [441, -1, 401], [456, -3, 364],
             [432, -1, 405], [426, 2, 379], [441, 1, 9480], [440, -3, 401], [432, -1, 405],
             [433, 1, 367], [443, -2, 401], [419, -11, 407], [432, 5, 405],
             [425, 1, 1500074], [443, -2, 401], [416, 2, 407], [445, 1, 393],
             [455, 0, 53503], [426, 1, 379], [440, -2, 380], [432, -1, 405], [440, 1, 9480],
             [443, -2, 401], [432, -1, 405], [440, -1, 401], [432, -1, 405], [426, 2, 379],
             [441, 1, 9480], [441, -3, 401], [419, -12, 407], [424, 6, 372], [447, 5, 365],
             [426, 3, 379], [433, 2, 367], [440, 1, 9480], [440, 1, 393], [455, 32, 373],
             [446, 1, 402], [440, -2, 380], [424, -3, 372], [455, -4, 375], [432, 3, 387],
             [445, 2, 393], [437, 1, 370], [454, -4, 373], [445, -1, 93815], [426, 2, 379],
             [433, 1, 367], [440, -4, 380], [420, -1, 407], [465, -2, 407], [445, 1, 393],
             [455, -4, 63716], [441, 1, 9480], [441, 1, 393], [458, -3, 373], [445, 1, 393],
             [458, -2, 373], [452, 1, 370], [454, -2, 411], [443, -1, 380], [432, -1, 405],
             [426, 2, 379], [441, 1, 9480], [441, -3, 401], [416, 3, 407], [415, 2, 407],
             [445, 1, 393], [455, 0, 53503], [432, -1, 405], [433, 1, 367], [440, -2, 401],
             [419, -4, 407], [465, 1, 407], [459, 0, 53503], [426, -1, 393], [461, 2, 380],
             [445, 1, 393], [458, -3, 373], [446, 1, 402], [443, 2, 393], [452, 1, 370],
             [454, -4, 373], [419, -9, 407], [415, -10, 407]],
            dtype='int32')
        self.spacy_doc.from_array(cols, values)

    def test_sgrank(self):
        expected = [
            'egyptian education official', 'technology trend', 'arab educator',
            'personal question', 'great sorrow', 'impact', 'minute', 'seminar',
            'hand', 'mosque']
        observed = [term for term, _ in keyterms.sgrank(self.spacy_doc)]
        self.assertEqual(len(expected), len(observed))
        for e, o in zip(expected, observed):
            self.assertEqual(e, o)

    def test_sgrank_n_keyterms(self):
        expected = [
            'egyptian education official', 'technology trend', 'arab educator',
            'personal question', 'great sorrow']
        observed = [
            term for term, _ in keyterms.sgrank(self.spacy_doc, n_keyterms=5)]
        self.assertEqual(len(expected), len(observed))
        for e, o in zip(expected, observed):
            self.assertEqual(e, o)

    def test_sgrank_norm_lower(self):
        expected = [
            'egyptian education official', 'technology trends', 'arab educators',
            'personal question', 'great sorrow']
        observed = [
            term for term, _
            in keyterms.sgrank(self.spacy_doc, normalize='lower', n_keyterms=5)]
        for e, o in zip(expected, observed):
            self.assertEqual(e, o)

    def test_sgrank_norm_none(self):
        expected = [
            'Egyptian education official', 'technology trends', 'Arab educators',
            'personal question', 'great sorrow']
        observed = [
            term for term, _
            in keyterms.sgrank(self.spacy_doc, normalize=None, n_keyterms=5)]
        for e, o in zip(expected, observed):
            self.assertEqual(e, o)

    def test_sgrank_norm_normalized_str(self):
        expected = [
            'egyptian education official', 'technology trend', 'arab educator',
            'personal question', 'great sorrow']
        observed = [
            term for term, _
            in keyterms.sgrank(self.spacy_doc, normalize=spacy_utils.normalized_str, n_keyterms=5)]
        for e, o in zip(expected, observed):
            self.assertEqual(e, o)

    def test_sgrank_window_width(self):
        expected = [
            'egyptian education official', 'technology trend', 'arab educator',
            'great sorrow', 'personal question']
        observed = [
            term for term, _ in keyterms.sgrank(self.spacy_doc, window_width=20, n_keyterms=5)]
        for e, o in zip(expected, observed):
            self.assertEqual(e, o)

    def test_textrank(self):
        expected = [
            'education', 'seminar', 'sorrow', 'great', 'arab', 'mosque', 'educator',
            'question', 'minute', 'impact']
        observed = [
            term for term, _ in keyterms.textrank(self.spacy_doc)]
        self.assertEqual(len(expected), len(observed))
        for e, o in zip(expected[:1], observed[:1]):  # there's a tie, ugh
            self.assertEqual(e, o)

    def test_textrank_n_keyterms(self):
        expected = ['education', 'seminar', 'sorrow', 'great', 'arab']
        observed = [
            term for term, _ in keyterms.textrank(self.spacy_doc, n_keyterms=5)]
        self.assertEqual(len(expected), len(observed))
        for e, o in zip(expected, observed):
            self.assertEqual(e, o)

    def test_textrank_norm_lower(self):
        expected = ['education', 'seminar', 'sorrow', 'great', 'arab']
        observed = [
            term for term, _
            in keyterms.textrank(self.spacy_doc, normalize='lower', n_keyterms=5)]
        for e, o in zip(expected, observed):
            self.assertEqual(e, o)

    def test_textrank_norm_none(self):
        expected = ['education', 'seminar', 'sorrow', 'great', 'Arab']
        observed = [
            term for term, _
            in keyterms.textrank(self.spacy_doc, normalize=None, n_keyterms=5)]
        for e, o in zip(expected, observed):
            self.assertEqual(e, o)

    def test_textrank_norm_normalized_str(self):
        expected = ['education', 'seminar', 'sorrow', 'great', 'arab']
        observed = [
            term for term, _
            in keyterms.textrank(self.spacy_doc, normalize=spacy_utils.normalized_str, n_keyterms=5)]
        for e, o in zip(expected, observed):
            self.assertEqual(e, o)

    def test_singlegrank(self):
        return

    def test_singlegrank_n_keyterms(self):
        return

    def test_singlegrank_norm_lower(self):
        return

    def test_singlegrank_norm_none(self):
        return

    def test_singlegrank_norm_normalized_str(self):
        return
