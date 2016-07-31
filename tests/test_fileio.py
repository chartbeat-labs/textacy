from __future__ import absolute_import, unicode_literals

import os
import tempfile
import unittest

import numpy as np
from scipy import sparse as sp
from spacy import attrs

from textacy import data, fileio
from textacy.compat import PY2, unicode_to_bytes


class FileIOTestCase(unittest.TestCase):

    def setUp(self):
        self.text = "The year was 2081, and everybody was finally equal. They weren't only equal before God and the law. They were equal every which way."
        self.spacy_pipeline = data.load_spacy('en')
        self.spacy_doc = self.spacy_pipeline(self.text)
        cols = [attrs.TAG, attrs.HEAD, attrs.DEP]
        values = np.array(
            [[426, 1, 379], [440, 1, 393], [455, 0, 53503], [425, -1, 369],
             [416, -2, 407], [424, -3, 372], [440, 1, 393], [455, -5, 375],
             [447, -1, 365], [433, -2, 363], [419, -3, 407], [445, 1, 393],
             [455, 0, 53503], [447, 2, 389], [447, 1, 365], [433, -3, 363],
             [432, -1, 405], [441, -1, 401], [424, -1, 372], [426, 1, 379],
             [440, -3, 375], [419, -9, 407], [445, 1, 393], [455, 0, 53503],
             [433, -1, 363], [426, 2, 379], [460, 1, 379], [440, -4, 392],
             [419, -5, 407]],
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
            fileio.write_file(expected, filename, mode='wb',
                              auto_make_dirs=True)
            observed = fileio.read_file(filename, mode='rb')
            self.assertEqual(observed, expected)

    def test_read_write_file_unicode(self):
        expected = self.text
        for ext in ('.txt', '.gz', '.bz2', '.xz'):
            filename = os.path.join(
                self.tempdir, 'test_read_write_file_unicode' + ext)
            if PY2 is True and ext != '.txt':
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
            if PY2 is True and ext != '.txt':
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
            if PY2 is True:
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
            if PY2 is True and ext != '.json':
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
            if PY2 is True:
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
            if PY2 is True and ext != '.json':
                self.assertRaises(
                    ValueError, fileio.open_sesame,
                    filename, 'wt', None, True)
            else:
                fileio.write_json_lines(expected, filename, mode='wt',
                                        auto_make_dirs=True)
                observed = list(fileio.read_json_lines(filename, mode='rt'))
                self.assertEqual(observed, expected)

    def test_read_write_spacy_docs(self):
        expected = [tok.lemma_ for tok in self.spacy_doc]
        for ext in ('.bin', '.bin.gz', '.bin.bz2', '.bin.xz'):
            filename = os.path.join(
                self.tempdir, 'test_read_write_spacy_docs' + ext)
            if PY2 is True and ext == '.bin.gz':  # no idea why this is the case
                self.assertRaises(
                    TypeError, fileio.write_spacy_docs,
                    self.spacy_doc, filename, True)
            else:
                fileio.write_spacy_docs(self.spacy_doc, filename, True)
                observed = [
                    tok.lemma_
                    for doc in fileio.read_spacy_docs(self.spacy_pipeline.vocab, filename)
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
        for fname in os.listdir(self.tempdir):
            os.remove(os.path.join(self.tempdir, fname))
        os.rmdir(self.tempdir)
