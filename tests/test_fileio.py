from __future__ import absolute_import, unicode_literals

import tempfile
import os
import unittest

import numpy as np
from spacy import attrs

from textacy import data, fileio


class FileIOTestCase(unittest.TestCase):

    def setUp(self):
        text = "The year was 2081, and everybody was finally equal. They weren't only equal before God and the law. They were equal every which way."
        self.spacy_pipeline = data.load_spacy_pipeline(lang='en')
        self.spacy_doc = self.spacy_pipeline(text)
        cols = [attrs.TAG, attrs.HEAD, attrs.DEP]
        values = np.array(
            [[426, 1, 379], [440, 1, 393], [455, 0, 53503], [425, -1, 369], [416, -2, 407],
            [424, -3, 372], [440, 1, 393], [455, -5, 375], [447, -1, 365], [433, -2, 363],
            [419, -3, 407], [445, 1, 393], [455, 0, 53503], [447, 2, 404], [447, -1, 365],
            [433, -3, 363], [432, -1, 405], [441, -1, 401], [424, -1, 372], [426, 1, 379],
            [440, -3, 375], [419, -9, 407], [445, 1, 393], [455, 0, 53503], [433, -1, 363],
            [426, 2, 379], [460, 1, 379], [440, -4, 392], [419, -5, 407]], dtype='int32')
        self.spacy_doc.from_array(cols, values)

    def test_write_conll(self):
        expected = "# sent_id 1\n1\tThe\tthe\tDET\tDT\t_\t2\tdet\t_\t_\n2\tyear\tyear\tNOUN\tNN\t_\t3\tnsubj\t_\t_\n3\twas\tbe\tVERB\tVBD\t_\t0\troot\t_\t_\n4\t2081\t2081\tNUM\tCD\t_\t3\tattr\t_\tSpaceAfter=No\n5\t,\t,\tPUNCT\t,\t_\t3\tpunct\t_\t_\n6\tand\tand\tCONJ\tCC\t_\t3\tcc\t_\t_\n7\teverybody\teverybody\tNOUN\tNN\t_\t8\tnsubj\t_\t_\n8\twas\tbe\tVERB\tVBD\t_\t3\tconj\t_\t_\n9\tfinally\tfinally\tADV\tRB\t_\t8\tadvmod\t_\t_\n10\tequal\tequal\tADJ\tJJ\t_\t8\tacomp\t_\tSpaceAfter=No\n11\t.\t.\tPUNCT\t.\t_\t8\tpunct\t_\t_\n\n# sent_id 2\n1\tThey\tthey\tNOUN\tPRP\t_\t2\tnsubj\t_\t_\n2\twere\tbe\tVERB\tVBD\t_\t0\troot\t_\tSpaceAfter=No\n3\tn't\tn't\tADV\tRB\t_\t5\tpreconj\t_\t_\n4\tonly\tonly\tADV\tRB\t_\t3\tadvmod\t_\t_\n5\tequal\tequal\tADJ\tJJ\t_\t2\tacomp\t_\t_\n6\tbefore\tbefore\tADP\tIN\t_\t5\tprep\t_\t_\n7\tGod\tgod\tNOUN\tNNP\t_\t6\tpobj\t_\t_\n8\tand\tand\tCONJ\tCC\t_\t7\tcc\t_\t_\n9\tthe\tthe\tDET\tDT\t_\t10\tdet\t_\t_\n10\tlaw\tlaw\tNOUN\tNN\t_\t7\tconj\t_\tSpaceAfter=No\n11\t.\t.\tPUNCT\t.\t_\t2\tpunct\t_\t_\n\n# sent_id 3\n1\tThey\tthey\tNOUN\tPRP\t_\t2\tnsubj\t_\t_\n2\twere\tbe\tVERB\tVBD\t_\t0\troot\t_\t_\n3\tequal\tequal\tADJ\tJJ\t_\t2\tacomp\t_\t_\n4\tevery\tevery\tDET\tDT\t_\t6\tdet\t_\t_\n5\twhich\twhich\tADJ\tWDT\t_\t6\tdet\t_\t_\n6\tway\tway\tNOUN\tNN\t_\t2\tnpadvmod\t_\tSpaceAfter=No\n7\t.\t.\tPUNCT\t.\t_\t2\tpunct\t_\tSpaceAfter=No\n"
        tempdir = tempfile.mkdtemp()
        filename = os.path.join(tempdir, 'test_write_conll.txt')
        fileio.write_conll(self.spacy_doc, filename)
        observed = fileio.read_file(filename)
        os.remove(filename)
        os.rmdir(tempdir)
        # nicer code below is only valid for Python 3.2 and later... sigh
        # with tempfile.TemporaryDirectory() as tempdir:
        #     filename = os.path.join(tempdir.title(), 'test_write_conll.txt')
        #     fileio.write_conll(self.spacy_doc, filename)
        #     observed = fileio.read_file(filename)
        self.assertEqual(observed, expected)

    def test_read_write_spacy_doc(self):
        expected = [tok.lemma_ for tok in self.spacy_doc]
        tempdir = tempfile.mkdtemp()
        filename = os.path.join(tempdir, 'test_read_write_spacy_doc.bin')
        fileio.write_spacy_docs(self.spacy_doc, filename)
        observed = [tok.lemma_ for doc in fileio.read_spacy_docs(self.spacy_pipeline.vocab, filename)
                    for tok in doc]
        os.remove(filename)
        os.rmdir(tempdir)
        self.assertEqual(observed, expected)

    def test_read_write_file_lines(self):
        expected = [sent.text for sent in self.spacy_doc.sents]
        tempdir = tempfile.mkdtemp()
        filename = os.path.join(tempdir, 'test_read_write_file_lines.txt')
        fileio.write_file_lines(expected, filename)
        observed = [line.strip() for line in fileio.read_file_lines(filename)]
        os.remove(filename)
        os.rmdir(tempdir)
        self.assertEqual(observed, expected)

    def test_read_write_file_lines_gzip(self):
        expected = [sent.text for sent in self.spacy_doc.sents]
        tempdir = tempfile.mkdtemp()
        filename = os.path.join(tempdir, 'test_read_write_file_lines.txt.gzip')
        fileio.write_file_lines(expected, filename)
        observed = [line.strip() for line in fileio.read_file_lines(filename)]
        os.remove(filename)
        os.rmdir(tempdir)
        self.assertEqual(observed, expected)

    def test_read_write_file_lines_bz2(self):
        expected = [sent.text for sent in self.spacy_doc.sents]
        tempdir = tempfile.mkdtemp()
        filename = os.path.join(tempdir, 'test_read_write_file_lines.txt.bz2')
        fileio.write_file_lines(expected, filename)
        observed = [line.strip() for line in fileio.read_file_lines(filename)]
        os.remove(filename)
        os.rmdir(tempdir)
        self.assertEqual(observed, expected)

    def test_read_write_json(self):
        expected = [{'idx': i, 'sent': sent.text}
                    for i, sent in enumerate(self.spacy_doc.sents)]
        tempdir = tempfile.mkdtemp()
        filename = os.path.join(tempdir, 'test_read_write_json.json')
        fileio.write_json(expected, filename)
        observed = list(fileio.read_json(filename, prefix=''))[0]
        os.remove(filename)
        os.rmdir(tempdir)
        self.assertEqual(observed, expected)

    def test_read_write_json_prefix(self):
        to_write = [{'idx': i, 'sent': sent.text}
                    for i, sent in enumerate(self.spacy_doc.sents)]
        expected = [item['sent'] for item in to_write]
        tempdir = tempfile.mkdtemp()
        filename = os.path.join(tempdir, 'test_read_write_json_prefix.json')
        fileio.write_json(to_write, filename)
        observed = list(fileio.read_json(filename, prefix='item.sent'))
        os.remove(filename)
        os.rmdir(tempdir)
        self.assertEqual(observed, expected)

    def test_read_write_json_lines(self):
        expected = [{'idx': i, 'sent': sent.text}
                    for i, sent in enumerate(self.spacy_doc.sents)]
        tempdir = tempfile.mkdtemp()
        filename = os.path.join(tempdir, 'test_read_write_json_lines.json')
        fileio.write_json_lines(expected, filename)
        observed = list(fileio.read_json_lines(filename))
        os.remove(filename)
        os.rmdir(tempdir)
        self.assertEqual(observed, expected)
