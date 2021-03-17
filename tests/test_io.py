import os
import tarfile
import zipfile

import numpy as np
import pytest
from scipy import sparse as sp

from textacy import load_spacy_lang
from textacy import io, utils

TEXT = (
    "The year was 2081, and everybody was finally equal. "
    "They weren't only equal before God and the law. "
    "They were equal every which way."
)
TESTS_DIR = os.path.split(__file__)[0]


@pytest.fixture(scope="module")
def spacy_doc():
    spacy_lang = load_spacy_lang("en_core_web_sm")
    spacy_doc = spacy_lang(TEXT)
    return spacy_doc


class TestTextIO:

    def test_read_write_bytes(self, tmpdir):
        expected = utils.to_bytes(TEXT)
        for ext in (".txt", ".gz", ".bz2", ".xz"):
            filepath = str(tmpdir.join("test_read_write_file_bytes" + ext))
            io.write_text(expected, filepath, mode="wb", make_dirs=True)
            observed = next(io.read_text(filepath, mode="rb"))
            assert observed == expected

    def test_read_write_unicode(self, tmpdir):
        expected = TEXT
        for ext in (".txt", ".gz", ".bz2", ".xz"):
            filepath = str(tmpdir.join("test_read_write_file_unicode" + ext))
            io.write_text(expected, filepath, mode="wt", make_dirs=True)
            observed = next(io.read_text(filepath, mode="rt"))
            assert observed == expected

    def test_read_write_bytes_lines(self, tmpdir, spacy_doc):
        expected = [utils.to_bytes(sent.text) for sent in spacy_doc.sents]
        for ext in (".txt", ".gz", ".bz2", ".xz"):
            filepath = str(tmpdir.join("test_read_write_file_lines_bytes" + ext))
            io.write_text(expected, filepath, mode="wb", make_dirs=True, lines=True)
            observed = [
                line.strip() for line in io.read_text(filepath, mode="rb", lines=True)
            ]
            assert observed == expected

    def test_read_write_unicode_lines(self, tmpdir, spacy_doc):
        expected = [sent.text for sent in spacy_doc.sents]
        for ext in (".txt", ".gz", ".bz2", ".xz"):
            filepath = str(tmpdir.join("test_read_write_file_lines_unicode" + ext))
            io.write_text(expected, filepath, mode="wt", make_dirs=True, lines=True)
            observed = [
                line.strip() for line in io.read_text(filepath, mode="rt", lines=True)
            ]
            assert observed == expected


class TestJSONIO:

    def test_read_write_bytes(self, tmpdir, spacy_doc):
        expected = [{"idx": i, "sent": sent.text} for i, sent in enumerate(spacy_doc.sents)]
        for ext in (".json", ".json.gz", ".json.bz2", ".json.xz"):
            filepath = str(tmpdir.join("test_read_write_json_bytes" + ext))
            with pytest.raises(TypeError):
                io.write_json(expected, filepath, "wb", make_dirs=True)

    def test_read_write_unicode(self, tmpdir, spacy_doc):
        expected = [{"idx": i, "sent": sent.text} for i, sent in enumerate(spacy_doc.sents)]
        for ext in (".json", ".json.gz", ".json.bz2", ".json.xz"):
            filepath = str(tmpdir.join("test_read_write_json_unicode" + ext))
            io.write_json(expected, filepath, mode="wt", make_dirs=True)
            observed = next(io.read_json(filepath, mode="rt", lines=False))
            assert observed == expected

    def test_read_write_bytes_lines(self, tmpdir, spacy_doc):
        expected = [{"idx": i, "sent": sent.text} for i, sent in enumerate(spacy_doc.sents)]
        for ext in (".json", ".json.gz", ".json.bz2", ".json.xz"):
            filepath = str(tmpdir.join("test_read_write_json_lines_bytes" + ext))
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
            io.write_json(expected, filepath, mode="wt", make_dirs=True, lines=True)
            observed = list(io.read_json(filepath, mode="rt", lines=True))
            assert observed == expected


class TestCSVIO:

    def test_read_write_compressed(self, tmpdir):
        expected = [
            ["this is some text", "scandal", 42.0],
            ["here's some more text: boom!", "escándalo", 1.0],
        ]
        for ext in (".csv", ".csv.gz", ".csv.bz2", ".csv.xz"):
            filepath = str(tmpdir.join("test_read_write_csv" + ext))
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


class TestSpacyIO:

    @pytest.mark.skip(reason="this takes wayyy too long now, reason unknown")
    def test_read_write_docs(self, tmpdir, spacy_doc):
        expected = [tok.lower_ for tok in spacy_doc]
        for ext in (".pkl", ".pkl.gz", ".pkl.bz2", ".pkl.xz"):
            filepath = str(tmpdir.join("test_read_write_spacy_docs" + ext))
            io.write_spacy_docs(spacy_doc, filepath, make_dirs=True)
            observed = [
                tok.lower_ for doc in io.read_spacy_docs(filepath) for tok in doc
            ]
            assert observed == expected

    def test_read_write_docs_binary(self, tmpdir, spacy_doc):
        expected = [tok.lower_ for tok in spacy_doc]
        filepath = str(tmpdir.join("test_read_write_spacy_docs_binary.bin"))
        io.write_spacy_docs(spacy_doc, filepath, make_dirs=True, format="binary")
        with pytest.raises(ValueError):
            next(io.read_spacy_docs(filepath, format="binary", lang=None))
        observed = [
            tok.lower_
            for doc in io.read_spacy_docs(filepath, format="binary", lang="en_core_web_sm")
            for tok in doc
        ]
        assert observed == expected

    def test_read_write_docs_binary_exclude(self, tmpdir, spacy_doc):
        expected = [tok.lower_ for tok in spacy_doc]
        filepath = str(tmpdir.join("test_read_write_spacy_docs_binary_exclude.bin"))
        io.write_spacy_docs(
            spacy_doc, filepath, make_dirs=True,
            format="binary", exclude=["sentiment", "user_data"],
        )
        observed = [
            tok.lower_
            for doc in io.read_spacy_docs(filepath, format="binary", lang="en_core_web_sm")
            for tok in doc
        ]
        assert observed == expected


class TestMatrixIO:

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


class TestIOUtils:

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

    def test_get_filename_from_url(self):
        url_fnames = [
            ["http://www.foo.bar/bat.zip", "bat.zip"],
            ["www.foo.bar/bat.tar.gz", "bat.tar.gz"],
            ["foo.bar/bat.zip?q=test", "bat.zip"],
            ["http%3A%2F%2Fwww.foo.bar%2Fbat.tar.gz", "bat.tar.gz"]
        ]
        for url, fname in url_fnames:
            assert io.get_filename_from_url(url) == fname

    def test_unpack_archive(self, tmpdir):
        data = "Here's some text data to pack and unpack."
        fpath_txt = str(tmpdir.join("test_unpack_archive.txt"))
        with io.open_sesame(fpath_txt, mode="wt") as f:
            f.write(data)
        fpath_zip = str(tmpdir.join("test_unpack_archive.zip"))
        with zipfile.ZipFile(fpath_zip, "w") as f:
            f.write(fpath_txt)
        io.unpack_archive(fpath_zip, extract_dir=tmpdir)
        fpath_tar = str(tmpdir.join("test_unpack_archive.tar"))
        with tarfile.TarFile(fpath_tar, "w") as f:
            f.add(fpath_txt)
        io.unpack_archive(fpath_tar, extract_dir=tmpdir)
        io.unpack_archive(fpath_txt, extract_dir=tmpdir)
