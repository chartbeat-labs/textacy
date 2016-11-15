from __future__ import absolute_import, unicode_literals

import numpy as np
import unittest

from spacy import attrs

from textacy import data, export


class ExportTestCase(unittest.TestCase):

    def setUp(self):
        text = "I would have lived in peace. But my enemies brought me war."
        # we're not loading all models for speed; instead, we're updating the doc
        # with pre-computed part-of-speech tagging and parsing values
        spacy_pipeline = data.load_spacy('en')
        self.spacy_doc = spacy_pipeline(text)
        cols = [attrs.TAG, attrs.HEAD, attrs.DEP]
        values = np.array(
            [[445, 3, 393], [437, 2, 370], [454, 1, 370], [457, 0, 53503],
             [432, -1, 405], [440, -1, 401], [419, -3, 407], [424, 3, 372],
             [446, 1, 402], [443, 1, 393], [455, 0, 53503], [445, -1, 93815],
             [440, -2, 380], [419, -3, 407]], dtype='int32')
        self.spacy_doc.from_array(cols, values)

    # def test_doc_to_gensim(self):
    #     expected_gdoc = [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]
    #     expected_gdict = {0: 'peace', 1: 'enemy', 2: 'war', 3: 'live', 4: 'bring'}
    #     observed_gdict, observed_gdoc = export.doc_to_gensim(
    #         self.spacy_doc, lemmatize=True,
    #         filter_stops=True, filter_punct=True, filter_nums=False)
    #     observed_gdict = dict(observed_gdict)
    #
    #     self.assertEqual(len(observed_gdoc), len(expected_gdoc))
    #     self.assertEqual(len(observed_gdict), len(expected_gdict))
    #     # ensure counts are the same for each unique token
    #     for exp_tok_id, exp_tok_str in expected_gdict.items():
    #         obs_tok_id = [tok_id for tok_id, tok_str in observed_gdict.items()
    #                       if tok_str == exp_tok_str][0]
    #         self.assertEqual(observed_gdoc[obs_tok_id][1], expected_gdoc[exp_tok_id][1])

    def test_write_conll(self):
        expected = '# sent_id 1\n1\tI\ti\tPRON\tPRP\t_\t4\tnsubj\t_\t_\n2\twould\twould\tVERB\tMD\t_\t4\taux\t_\t_\n3\thave\thave\tVERB\tVB\t_\t4\taux\t_\t_\n4\tlived\tlive\tVERB\tVBN\t_\t0\troot\t_\t_\n5\tin\tin\tADP\tIN\t_\t4\tprep\t_\t_\n6\tpeace\tpeace\tNOUN\tNN\t_\t5\tpobj\t_\tSpaceAfter=No\n7\t.\t.\tPUNCT\t.\t_\t4\tpunct\t_\t_\n\n# sent_id 2\n1\tBut\tbut\tCONJ\tCC\t_\t4\tcc\t_\t_\n2\tmy\tmy\tADJ\tPRP$\t_\t3\tposs\t_\t_\n3\tenemies\tenemy\tNOUN\tNNS\t_\t4\tnsubj\t_\t_\n4\tbrought\tbring\tVERB\tVBD\t_\t0\troot\t_\t_\n5\tme\tme\tPRON\tPRP\t_\t4\tdative\t_\t_\n6\twar\twar\tNOUN\tNN\t_\t4\tdobj\t_\tSpaceAfter=No\n7\t.\t.\tPUNCT\t.\t_\t4\tpunct\t_\tSpaceAfter=No\n'
        observed = export.doc_to_conll(self.spacy_doc)
        self.assertEqual(observed, expected)
