# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

import os
import shutil
import tempfile
import unittest

import numpy as np
from scipy import sparse as sp
from spacy import attrs

from textacy import data, fileio
from textacy.compat import is_python2, unicode_to_bytes


class FileIOTestCase(unittest.TestCase):

    def setUp(self):
        self.text = "The year was 2081, and everybody was finally equal. They weren't only equal before God and the law. They were equal every which way."
        self.spacy_lang = data.load_spacy('en')
        self.spacy_doc = self.spacy_lang(self.text)
        cols = [attrs.TAG, attrs.HEAD, attrs.DEP]
        values = np.array(
            [[460, 1, 411], [474, 1, 425], [489, 0, 512817], [459, -1, 399],
             [450, -1, 441], [458, -2, 403], [474, 1, 425], [489, -4, 406],
             [481, 1, 396], [467, -2, 394], [453, -8, 441], [479, 1, 425],
             [489, 0, 512817], [481, 2, 438], [481, 1, 396], [467, -3, 394],
             [466, -1, 439], [475, -1, 435], [458, -1, 403], [460, 1, 411],
             [474, -5, 412], [453, -9, 441], [479, 1, 425], [489, 0, 512817],
             [467, -1, 394], [460, 2, 411], [494, 1, 411], [474, -3, 758141],
             [453, -5, 441]],
            dtype='int32')
        self.spacy_doc.from_array(cols, values)
        self.tempdir = tempfile.mkdtemp(
            prefix='test_fileio', dir=os.path.dirname(os.path.abspath(__file__)))
        self.tests_dir = os.path.split(__file__)[0]
        self.maxDiff = None

    def test_read_write_file_bytes(self):
        expected = unicode_to_bytes(self.text)
        for ext in ('.txt', '.gz', '.bz2', '.xz'):
            filename = os.path.join(
                self.tempdir, 'test_read_write_file_bytes' + ext)
            if is_python2 is True and ext == '.xz':
                self.assertRaises(
                    ValueError, fileio.open_sesame,
                    filename, 'wb', 'utf-8', True)
            else:
                fileio.write_file(expected, filename, mode='wb',
                                  auto_make_dirs=True)
                observed = fileio.read_file(filename, mode='rb')
                self.assertEqual(observed, expected)

    def test_read_write_file_unicode(self):
        expected = self.text
        for ext in ('.txt', '.gz', '.bz2', '.xz'):
            filename = os.path.join(
                self.tempdir, 'test_read_write_file_unicode' + ext)
            if is_python2 is True and ext != '.txt':
                self.assertRaises(
                    ValueError, fileio.open_sesame,
                    filename, 'wt', 'utf-8', True)
            else:
                fileio.write_file(expected, filename, mode='wt',
                                  auto_make_dirs=True)
                observed = fileio.read_file(filename, mode='rt')
                self.assertEqual(observed, expected)

    def test_read_write_file_lines_bytes(self):
        expected = [unicode_to_bytes(sent.text) for sent in self.spacy_doc.sents]
        for ext in ('.txt', '.gz', '.bz2', '.xz'):
            filename = os.path.join(
                self.tempdir, 'test_read_write_file_lines_bytes' + ext)
            if is_python2 is True and ext == '.xz':
                self.assertRaises(
                    ValueError, fileio.open_sesame,
                    filename, 'wb', 'utf-8', True)
            else:
                fileio.write_file_lines(expected, filename, mode='wb',
                                        auto_make_dirs=True)
                observed = [line.strip() for line
                            in fileio.read_file_lines(filename, mode='rb')]
                self.assertEqual(observed, expected)

    def test_read_write_file_lines_unicode(self):
        expected = [sent.text for sent in self.spacy_doc.sents]
        for ext in ('.txt', '.gz', '.bz2', '.xz'):
            filename = os.path.join(
                self.tempdir, 'test_read_write_file_lines_unicode' + ext)
            if is_python2 is True and ext != '.txt':
                self.assertRaises(
                    ValueError, fileio.open_sesame,
                    filename, 'wt', None, True)
            else:
                fileio.write_file_lines(expected, filename, mode='wt',
                                        auto_make_dirs=True)
                observed = [line.strip() for line
                            in fileio.read_file_lines(filename, mode='rt')]
                self.assertEqual(observed, expected)

    def test_read_write_json_bytes(self):
        expected = [{'idx': i, 'sent': sent.text}
                    for i, sent in enumerate(self.spacy_doc.sents)]
        for ext in ('.json', '.json.gz', '.json.bz2', '.json.xz'):
            filename = os.path.join(
                self.tempdir, 'test_read_write_json_bytes' + ext)
            if is_python2 is True:
                if ext == '.json.xz':
                    self.assertRaises(
                        ValueError, fileio.open_sesame,
                        filename, 'wb', 'utf-8', True)
                else:
                    fileio.write_json(expected, filename, mode='wb',
                                      auto_make_dirs=True)
                    observed = list(fileio.read_json(filename, mode='rb', prefix=''))[0]
                    self.assertEqual(observed, expected)
            else:
                self.assertRaises(
                    TypeError,
                    lambda: fileio.write_json(expected, filename, 'wb',
                                              auto_make_dirs=True))

    def test_read_write_json_unicode(self):
        expected = [{'idx': i, 'sent': sent.text}
                    for i, sent in enumerate(self.spacy_doc.sents)]
        for ext in ('.json', '.json.gz', '.json.bz2', '.json.xz'):
            filename = os.path.join(
                self.tempdir, 'test_read_write_json_unicode' + ext)
            if is_python2 is True and ext != '.json':
                self.assertRaises(
                    ValueError, fileio.open_sesame,
                    filename, 'wt', None, True)
            else:
                fileio.write_json(expected, filename, mode='wt',
                                  auto_make_dirs=True)
                observed = list(fileio.read_json(filename, mode='rt', prefix=''))[0]
                self.assertEqual(observed, expected)

    def test_read_write_json_prefix(self):
        to_write = [{'idx': i, 'sent': sent.text}
                    for i, sent in enumerate(self.spacy_doc.sents)]
        for prefix in ('idx', 'sent'):
            expected = [item[prefix] for item in to_write]
            filename = os.path.join(
                self.tempdir, 'test_read_write_json_prefix.json')
            fileio.write_json(to_write, filename, auto_make_dirs=True)
            observed = list(fileio.read_json(filename, prefix='item.' + prefix))
            self.assertEqual(observed, expected)

    def test_read_write_json_lines_bytes(self):
        expected = [{'idx': i, 'sent': sent.text}
                    for i, sent in enumerate(self.spacy_doc.sents)]
        for ext in ('.json', '.json.gz', '.json.bz2', '.json.xz'):
            filename = os.path.join(
                self.tempdir, 'test_read_write_json_lines_bytes' + ext)
            if is_python2 is True:
                if ext == '.json.xz':
                    self.assertRaises(
                        ValueError, fileio.open_sesame,
                        filename, 'wb', 'utf-8', True)
                else:
                    fileio.write_json_lines(expected, filename, mode='wb',
                                            auto_make_dirs=True)
                    observed = list(fileio.read_json_lines(filename, mode='rb'))
                    self.assertEqual(observed, expected)
            else:
                self.assertRaises(
                    TypeError, fileio.write_json_lines,
                    expected, filename, 'wb', None, True)

    def test_read_write_json_lines_unicode(self):
        expected = [{'idx': i, 'sent': sent.text}
                    for i, sent in enumerate(self.spacy_doc.sents)]
        for ext in ('.json', '.json.gz', '.json.bz2', '.json.xz'):
            filename = os.path.join(
                self.tempdir, 'test_read_write_json_lines_unicode' + ext)
            if is_python2 is True and ext != '.json':
                self.assertRaises(
                    ValueError, fileio.open_sesame,
                    filename, 'wt', None, True)
            else:
                fileio.write_json_lines(expected, filename, mode='wt',
                                        auto_make_dirs=True)
                observed = list(fileio.read_json_lines(filename, mode='rt'))
                self.assertEqual(observed, expected)

    def test_read_write_csv_compressed(self):
        expected = [['this is some text', 'scandal', '42'],
                    ["here's some more text: boom!", 'escándalo', '1.0']]
        for ext in ('.csv', '.csv.gz', '.csv.bz2', '.csv.xz'):
            filename = os.path.join(
                self.tempdir, 'test_read_write_csv' + ext)
            if is_python2 is True and ext != '.csv':
                self.assertRaises(
                    ValueError, fileio.open_sesame,
                    filename, 'wt', None, True)
            else:
                fileio.write_csv(expected, filename, auto_make_dirs=True)
                observed = list(fileio.read_csv(filename))
                self.assertEqual(observed, expected)

    def test_read_write_csv_delimiters(self):
        expected = [['this is some text', 'scandal', '42'],
                    ["here's some more text: boom!", 'escándalo', '1.0']]
        for delimiter in (',', '\t', '|', ':'):
            filename = os.path.join(
                self.tempdir, 'test_read_write_csv.csv')
            fileio.write_csv(
                expected, filename, delimiter=delimiter, auto_make_dirs=True)
            observed = list(fileio.read_csv(filename, delimiter=delimiter))
            self.assertEqual(observed, expected)

    def test_read_write_csv_dialect(self):
        expected = [['this is some text', 'scandal', '42'],
                    ["here's some more text: boom!", 'escándalo', '1.0']]
        filename = os.path.join(
            self.tempdir, 'test_read_write_csv.csv')
        fileio.write_csv(
            expected, filename, dialect='excel', auto_make_dirs=True)
        observed = list(fileio.read_csv(filename, dialect='infer'))
        self.assertEqual(observed, expected)

    def test_read_write_spacy_docs(self):
        expected = [tok.lemma_ for tok in self.spacy_doc]
        for ext in ('.bin', '.bin.gz', '.bin.bz2', '.bin.xz'):
            filename = os.path.join(
                self.tempdir, 'test_read_write_spacy_docs' + ext)
            if is_python2 is True and ext == '.bin.xz':
                self.assertRaises(
                    ValueError, fileio.open_sesame,
                    filename, 'wb', None, True)
            elif is_python2 is True and ext == '.bin.gz':  # no idea why this is the case
                self.assertRaises(
                    TypeError, fileio.write_spacy_docs,
                    self.spacy_doc, filename, True)
            else:
                fileio.write_spacy_docs(self.spacy_doc, filename, True)
                observed = [
                    tok.lemma_
                    for doc in fileio.read_spacy_docs(self.spacy_lang.vocab, filename)
                    for tok in doc]
                self.assertEqual(observed, expected)

    def test_read_write_sparse_csr_matrix(self):
        expected = sp.csr_matrix(
            (np.array([1, 2, 3, 4, 5, 6]),
             (np.array([0, 0, 1, 2, 2, 2]), np.array([0, 2, 2, 0, 1, 2]))),
            shape=(3, 3))
        filename = os.path.join(self.tempdir, 'test_read_write_sparse_csr_matrix.npz')
        fileio.write_sparse_matrix(expected, filename, compressed=False)
        observed = fileio.read_sparse_csr_matrix(filename)
        self.assertEqual(abs(observed - expected).nnz, 0)

    def test_read_write_sparse_csr_matrix_compressed(self):
        expected = sp.csr_matrix(
            (np.array([1, 2, 3, 4, 5, 6]),
             (np.array([0, 0, 1, 2, 2, 2]), np.array([0, 2, 2, 0, 1, 2]))),
            shape=(3, 3))
        filename = os.path.join(self.tempdir, 'test_read_write_sparse_csr_matrix_compressed.npz')
        fileio.write_sparse_matrix(expected, filename, compressed=True)
        observed = fileio.read_sparse_csr_matrix(filename)
        self.assertEqual(abs(observed - expected).nnz, 0)

    def test_read_write_sparse_csc_matrix(self):
        expected = sp.csc_matrix(
            (np.array([1, 2, 3, 4, 5, 6]),
             (np.array([0, 0, 1, 2, 2, 2]), np.array([0, 2, 2, 0, 1, 2]))),
            shape=(3, 3))
        filename = os.path.join(self.tempdir, 'test_read_write_sparse_csc_matrix.npz')
        fileio.write_sparse_matrix(expected, filename, compressed=False)
        observed = fileio.read_sparse_csc_matrix(filename)
        self.assertEqual(abs(observed - expected).nnz, 0)

    def test_read_write_sparse_csc_matrix_compressed(self):
        expected = sp.csc_matrix(
            (np.array([1, 2, 3, 4, 5, 6]),
             (np.array([0, 0, 1, 2, 2, 2]), np.array([0, 2, 2, 0, 1, 2]))),
            shape=(3, 3))
        filename = os.path.join(self.tempdir, 'test_read_write_sparse_csc_matrix_compressed.npz')
        fileio.write_sparse_matrix(expected, filename, compressed=True)
        observed = fileio.read_sparse_csc_matrix(filename)
        self.assertEqual(abs(observed - expected).nnz, 0)

    def test_get_filenames(self):
        expected = sorted(os.path.join(self.tests_dir, fname)
                          for fname in os.listdir(self.tests_dir)
                          if os.path.isfile(os.path.join(self.tests_dir, fname)))
        observed = sorted(fileio.get_filenames(self.tests_dir,
                                               ignore_invisible=False,
                                               recursive=False))
        self.assertEqual(observed, expected)

    def test_get_filenames_ignore_invisible(self):
        path = os.path.dirname(os.path.abspath(__file__))
        self.assertTrue(
            len(list(fileio.get_filenames(path, ignore_invisible=True))) <=
            len(list(fileio.get_filenames(path, ignore_invisible=False)))
            )

    def test_get_filenames_ignore_regex(self):
        self.assertTrue(
            len(list(fileio.get_filenames(self.tests_dir,
                                          ignore_regex='test_',
                                          ignore_invisible=True))) == 0)

    def test_get_filenames_match_regex(self):
        self.assertTrue(
            len(list(fileio.get_filenames(self.tests_dir,
                                          match_regex='fileio',
                                          extension='.py'))) == 1)

    def tearDown(self):
        shutil.rmtree(self.tempdir)
