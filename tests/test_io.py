# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

import os

import numpy as np
import pytest
from scipy import sparse as sp

from textacy import cache, compat, io

TEXT = (
    "The year was 2081, and everybody was finally equal. "
    "They weren't only equal before God and the law. "
    "They were equal every which way."
)
TESTS_DIR = os.path.split(__file__)[0]


@pytest.fixture(scope="module")
def spacy_doc():
    spacy_lang = cache.load_spacy_lang("en")
    spacy_doc = spacy_lang(TEXT)
    return spacy_doc


class TestTextIO(object):

    def test_read_write_bytes(self, tmpdir):
        expected = compat.to_bytes(TEXT)
        for ext in (".txt", ".gz", ".bz2", ".xz"):
            filepath = str(tmpdir.join("test_read_write_file_bytes" + ext))
            if compat.PY2 is True and ext == ".xz":
                with pytest.raises(ValueError):
                    io.open_sesame(filepath, mode="wb", encoding="utf-8", make_dirs=True)
            else:
                io.write_text(expected, filepath, mode="wb", make_dirs=True)
                observed = next(io.read_text(filepath, mode="rb"))
                assert observed == expected

    def test_read_write_unicode(self, tmpdir):
        expected = TEXT
        for ext in (".txt", ".gz", ".bz2", ".xz"):
            filepath = str(tmpdir.join("test_read_write_file_unicode" + ext))
            if compat.PY2 is True and ext != ".txt":
                with pytest.raises(ValueError):
                    io.open_sesame(filepath, mode="wt", encoding="utf-8", make_dirs=True)
            else:
                io.write_text(expected, filepath, mode="wt", make_dirs=True)
                observed = next(io.read_text(filepath, mode="rt"))
                assert observed == expected

    def test_read_write_bytes_lines(self, tmpdir, spacy_doc):
        expected = [compat.to_bytes(sent.text) for sent in spacy_doc.sents]
        for ext in (".txt", ".gz", ".bz2", ".xz"):
            filepath = str(tmpdir.join("test_read_write_file_lines_bytes" + ext))
            if compat.PY2 is True and ext == ".xz":
                with pytest.raises(ValueError):
                    io.open_sesame(filepath, mode="wb", encoding="utf-8", make_dirs=True)
            else:
                io.write_text(expected, filepath, mode="wb", make_dirs=True, lines=True)
                observed = [
                    line.strip() for line in io.read_text(filepath, mode="rb", lines=True)
                ]
                assert observed == expected

    def test_read_write_unicode_lines(self, tmpdir, spacy_doc):
        expected = [sent.text for sent in spacy_doc.sents]
        for ext in (".txt", ".gz", ".bz2", ".xz"):
            filepath = str(tmpdir.join("test_read_write_file_lines_unicode" + ext))
            if compat.PY2 is True and ext != ".txt":
                with pytest.raises(ValueError):
                    io.open_sesame(filepath, mode="wt", encoding=None, make_dirs=True)
            else:
                io.write_text(expected, filepath, mode="wt", make_dirs=True, lines=True)
                observed = [
                    line.strip() for line in io.read_text(filepath, mode="rt", lines=True)
                ]
                assert observed == expected


class TestJSONIO(object):

    def test_read_write_bytes(self, tmpdir, spacy_doc):
        expected = [{"idx": i, "sent": sent.text} for i, sent in enumerate(spacy_doc.sents)]
        for ext in (".json", ".json.gz", ".json.bz2", ".json.xz"):
            filepath = str(tmpdir.join("test_read_write_json_bytes" + ext))
            if compat.PY2 is True:
                if ext == ".json.xz":
                    with pytest.raises(ValueError):
                        io.open_sesame(
                            filepath, mode="wb", encoding="utf-8", make_dirs=True
                        )
                else:
                    io.write_json(expected, filepath, mode="wb", make_dirs=True)
                    observed = next(io.read_json(filepath, mode="rb", lines=False))
                    assert observed == expected
            else:
                with pytest.raises(TypeError):
                    io.write_json(expected, filepath, "wb", make_dirs=True)

    def test_read_write_unicode(self, tmpdir, spacy_doc):
        expected = [{"idx": i, "sent": sent.text} for i, sent in enumerate(spacy_doc.sents)]
        for ext in (".json", ".json.gz", ".json.bz2", ".json.xz"):
            filepath = str(tmpdir.join("test_read_write_json_unicode" + ext))
            if compat.PY2 is True and ext != ".json":
                with pytest.raises(ValueError):
                    io.open_sesame(filepath, mode="wt", encoding=None, make_dirs=True)
            else:
                io.write_json(expected, filepath, mode="wt", make_dirs=True)
                observed = next(io.read_json(filepath, mode="rt", lines=False))
                assert observed == expected

    def test_read_write_bytes_lines(self, tmpdir, spacy_doc):
        expected = [{"idx": i, "sent": sent.text} for i, sent in enumerate(spacy_doc.sents)]
        for ext in (".json", ".json.gz", ".json.bz2", ".json.xz"):
            filepath = str(tmpdir.join("test_read_write_json_lines_bytes" + ext))
            if compat.PY2 is True:
                if ext == ".json.xz":
                    with pytest.raises(ValueError):
                        io.open_sesame(
                            filepath, mode="wb", encoding="utf-8", make_dirs=True
                        )
                else:
                    io.write_json(expected, filepath, mode="wb", make_dirs=True, lines=True)
                    observed = list(io.read_json(filepath, mode="rb", lines=True))
                    assert observed == expected
            else:
                with pytest.raises(TypeError):
                    io.write_json(
                        expected,
                        filepath,
                        mode="wb",
                        encoding=None,
                        make_dirs=True,
                        lines=True,
                    )

    def test_read_write_unicode_lines(self, tmpdir, spacy_doc):
        expected = [{"idx": i, "sent": sent.text} for i, sent in enumerate(spacy_doc.sents)]
        for ext in (".json", ".json.gz", ".json.bz2", ".json.xz"):
            filepath = str(tmpdir.join("test_read_write_json_lines_unicode" + ext))
            if compat.PY2 is True and ext != ".json":
                with pytest.raises(ValueError):
                    io.open_sesame(filepath, mode="wt", encoding=None, make_dirs=True)
            else:
                io.write_json(expected, filepath, mode="wt", make_dirs=True, lines=True)
                observed = list(io.read_json(filepath, mode="rt", lines=True))
                assert observed == expected


class TestCSVIO(object):

    def test_read_write_compressed(self, tmpdir):
        expected = [
            ["this is some text", "scandal", 42.0],
            ["here's some more text: boom!", "escándalo", 1.0],
        ]
        for ext in (".csv", ".csv.gz", ".csv.bz2", ".csv.xz"):
            filepath = str(tmpdir.join("test_read_write_csv" + ext))
            if compat.PY2 is True and ext != ".csv":
                with pytest.raises(ValueError):
                    io.open_sesame(filepath, mode="wt", encoding=None, make_dirs=True)
            else:
                io.write_csv(expected, filepath, make_dirs=True)
                observed = list(io.read_csv(filepath))
                assert observed == expected

    def test_read_write_delimiters(self, tmpdir):
        expected = [
            ["this is some text", "scandal", 42.0],
            ["here's some more text: boom!", "escándalo", 1.0],
        ]
        for delimiter in (",", "\t", "|", ":"):
            filepath = str(tmpdir.join("test_read_write_csv.csv"))
            io.write_csv(expected, filepath, delimiter=delimiter, make_dirs=True)
            observed = list(io.read_csv(filepath, delimiter=delimiter))
            assert observed == expected

    def test_read_write_dialect(self, tmpdir):
        expected = [
            ["this is some text", "scandal", 42.0],
            ["here's some more text: boom!", "escándalo", 1.0],
        ]
        filepath = str(tmpdir.join("test_read_write_csv.csv"))
        io.write_csv(expected, filepath, dialect="excel", make_dirs=True)
        observed = list(io.read_csv(filepath, dialect="infer"))
        assert observed == expected

    def test_read_write_dict(self, tmpdir):
        expected = [
            {"text": "this is some text", "kind": "scandal", "number": 42.0},
            {"text": "here's some more text: boom!", "kind": "escándalo", "number": 1.0},
        ]
        filepath = str(tmpdir.join("test_read_write_csv_dict.csv"))
        io.write_csv(
            expected,
            filepath,
            dialect="excel",
            make_dirs=True,
            fieldnames=["text", "kind", "number"],
        )
        observed = [
            dict(item)
            for item in io.read_csv(
                filepath, dialect="excel", fieldnames=["text", "kind", "number"]
            )
        ]
        assert observed == expected


class TestSpacyIO(object):

    def test_read_write_docs(self, tmpdir, spacy_doc):
        expected = [tok.lower_ for tok in spacy_doc]
        for ext in (".pkl", ".pkl.gz", ".pkl.bz2", ".pkl.xz"):
            filepath = str(tmpdir.join("test_read_write_spacy_docs" + ext))
            if compat.PY2 is True and ext == ".pkl.xz":
                with pytest.raises(ValueError):
                    io.open_sesame(filepath, mode="wb", encoding=None, make_dirs=True)
            else:
                io.write_spacy_docs(spacy_doc, filepath, True)
                observed = [
                    tok.lower_ for doc in io.read_spacy_docs(filepath) for tok in doc
                ]
                assert observed == expected

    def test_read_write_docs_binary(self, tmpdir, spacy_doc):
        expected = [tok.lower_ for tok in spacy_doc]
        filepath = str(tmpdir.join("test_read_write_spacy_docs_binary.bin"))
        io.write_spacy_docs(spacy_doc, filepath, True, format="binary")
        with pytest.raises(ValueError):
            next(io.read_spacy_docs(filepath, format="binary", lang=None))
        observed = [
            tok.lower_
            for doc in io.read_spacy_docs(filepath, format="binary", lang="en")
            for tok in doc
        ]
        assert observed == expected

    def test_read_write_docs_binary_exclude(self, tmpdir, spacy_doc):
        expected = [tok.lower_ for tok in spacy_doc]
        filepath = str(tmpdir.join("test_read_write_spacy_docs_binary_exclude.bin"))
        io.write_spacy_docs(
            spacy_doc, filepath, True,
            format="binary", exclude=["sentiment", "user_data"],
        )
        observed = [
            tok.lower_
            for doc in io.read_spacy_docs(filepath, format="binary", lang="en")
            for tok in doc
        ]
        assert observed == expected


class TestMatrixIO(object):

    def test_read_write_sparse_csr(self, tmpdir):
        expected = sp.csr_matrix(
            (
                np.array([1, 2, 3, 4, 5, 6]),
                (np.array([0, 0, 1, 2, 2, 2]), np.array([0, 2, 2, 0, 1, 2])),
            ),
            shape=(3, 3),
        )
        filepath = str(tmpdir.join("test_read_write_sparse_matrix_csr.npz"))
        io.write_sparse_matrix(expected, filepath, compressed=False)
        observed = io.read_sparse_matrix(filepath, kind="csr")
        assert abs(observed - expected).nnz == 0

    def test_read_write_sparse_csr_compressed(self, tmpdir):
        expected = sp.csr_matrix(
            (
                np.array([1, 2, 3, 4, 5, 6]),
                (np.array([0, 0, 1, 2, 2, 2]), np.array([0, 2, 2, 0, 1, 2])),
            ),
            shape=(3, 3),
        )
        filepath = str(tmpdir.join("test_read_write_sparse_matrix_csr_compressed.npz"))
        io.write_sparse_matrix(expected, filepath, compressed=True)
        observed = io.read_sparse_matrix(filepath, kind="csr")
        assert abs(observed - expected).nnz == 0

    def test_read_write_sparse_csc(self, tmpdir):
        expected = sp.csc_matrix(
            (
                np.array([1, 2, 3, 4, 5, 6]),
                (np.array([0, 0, 1, 2, 2, 2]), np.array([0, 2, 2, 0, 1, 2])),
            ),
            shape=(3, 3),
        )
        filepath = str(tmpdir.join("test_read_write_sparse_matrix_csc.npz"))
        io.write_sparse_matrix(expected, filepath, compressed=False)
        observed = io.read_sparse_matrix(filepath, kind="csc")
        assert abs(observed - expected).nnz == 0

    def test_read_write_sparse_csc_compressed(self, tmpdir):
        expected = sp.csc_matrix(
            (
                np.array([1, 2, 3, 4, 5, 6]),
                (np.array([0, 0, 1, 2, 2, 2]), np.array([0, 2, 2, 0, 1, 2])),
            ),
            shape=(3, 3),
        )
        filepath = str(tmpdir.join("test_read_write_sparse_matrix_csc_compressed.npz"))
        io.write_sparse_matrix(expected, filepath, compressed=True)
        observed = io.read_sparse_matrix(filepath, kind="csc")
        assert abs(observed - expected).nnz == 0


class TestIOUtils(object):

    def test_get_filepaths(self):
        expected = sorted(
            os.path.join(TESTS_DIR, fname)
            for fname in os.listdir(TESTS_DIR)
            if os.path.isfile(os.path.join(TESTS_DIR, fname))
        )
        observed = sorted(
            io.get_filepaths(TESTS_DIR, ignore_invisible=False, recursive=False)
        )
        assert observed == expected

    def test_get_filepaths_ignore_invisible(self):
        dirpath = os.path.dirname(os.path.abspath(__file__))
        assert len(list(io.get_filepaths(dirpath, ignore_invisible=True))) <= len(
            list(io.get_filepaths(dirpath, ignore_invisible=False))
        )

    def test_get_filepaths_ignore_regex(self):
        assert (
            len(
                list(
                    io.get_filepaths(TESTS_DIR, ignore_regex="test_", ignore_invisible=True)
                )
            )
            == 0
        )

    def test_get_filepaths_match_regex(self):
        assert (
            len(list(io.get_filepaths(TESTS_DIR, match_regex="io", extension=".py"))) == 1
        )

    def test_get_filenames(self):
        expected = sorted(
            os.path.join(TESTS_DIR, fname)
            for fname in os.listdir(TESTS_DIR)
            if os.path.isfile(os.path.join(TESTS_DIR, fname))
        )
        observed = sorted(
            io.get_filenames(TESTS_DIR, ignore_invisible=False, recursive=False)
        )
        assert observed == expected
