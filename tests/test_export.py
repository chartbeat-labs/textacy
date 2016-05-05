from __future__ import absolute_import, unicode_literals

import numpy as np
import unittest

from spacy import attrs

from textacy import data, export

class ExportTestCase(unittest.TestCase):

    def setUp(self):
        text = "The year was 2081, and everybody was finally equal. They weren't only equal before God and the law. They were equal every which way."
        # we're not loading all models for speed; instead, we're updating the doc
        # with pre-computed part-of-speech tagging and parsing values
        spacy_pipeline = data.load_spacy('en')
        self.spacy_doc = spacy_pipeline(text)
        cols = [attrs.TAG, attrs.HEAD, attrs.DEP]
        values = np.array(
            [[426, 1, 379], [440, 1, 393], [455, 0, 53503], [425, -1, 369], [416, -2, 407],
            [424, -3, 372], [440, 1, 393], [455, -5, 375], [447, -1, 365], [433, -2, 363],
            [419, -3, 407], [445, 1, 393], [455, 0, 53503], [447, 2, 404], [447, -1, 365],
            [433, -3, 363], [432, -1, 405], [441, -1, 401], [424, -1, 372], [426, 1, 379],
            [440, -3, 375], [419, -9, 407], [445, 1, 393], [455, 0, 53503], [433, -1, 363],
            [426, 2, 379], [460, 1, 379], [440, -4, 392], [419, -5, 407]], dtype='int32')
        self.spacy_doc.from_array(cols, values)

    def test_doc_to_gensim(self):
        expected_gdoc = [(0, 1), (1, 1), (2, 1), (3, 3), (4, 1), (5, 1), (6, 1),
                         (7, 1), (8, 1)]
        expected_gdict = {0: 'year', 1: '2081', 2: 'law', 3: 'equal', 4: 'finally',
                          5: 'way', 6: 'everybody', 7: "n't", 8: 'god'}
        observed_gdict, observed_gdoc = export.doc_to_gensim(
            self.spacy_doc, lemmatize=True,
            filter_stops=True, filter_punct=True, filter_nums=False)
        observed_gdict = dict(observed_gdict)

        self.assertEqual(len(observed_gdoc), len(expected_gdoc))
        self.assertEqual(len(observed_gdict), len(expected_gdict))
        # ensure counts are the same for each unique token
        for exp_tok_id, exp_tok_str in expected_gdict.items():
            obs_tok_id = [tok_id for tok_id, tok_str in observed_gdict.items()
                          if tok_str == exp_tok_str][0]
            self.assertEqual(observed_gdoc[obs_tok_id][1], expected_gdoc[exp_tok_id][1])
