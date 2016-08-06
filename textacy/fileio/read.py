"""
Module with functions for reading content from disk in common formats.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from functools import partial
import json

import ijson
from numpy import load as np_load
from scipy.sparse import csc_matrix, csr_matrix
from spacy.tokens.doc import Doc as SpacyDoc

from textacy.compat import csv
from textacy.fileio import open_sesame

JSON_DECODER = json.JSONDecoder()


def read_file(filepath, mode='rt', encoding=None):
    """
    Read the full contents of a file. Files compressed with gzip, bz2, or lzma
    are handled automatically.
    """
    with open_sesame(filepath, mode=mode, encoding=encoding) as f:
        return f.read()


def read_file_lines(filepath, mode='rt', encoding=None):
    """
    Read the contents of a file, line by line. Files compressed with gzip, bz2,
    or lzma are handled automatically.
    """
    with open_sesame(filepath, mode=mode, encoding=encoding) as f:
        for line in f:
            yield line


def read_json(filepath, mode='rt', encoding=None, prefix=''):
    """
    Iterate over JSON objects matching the field given by ``prefix``.
    Useful for reading a large JSON array one item (with ``prefix='item'``)
    or sub-item (``prefix='item.fieldname'``) at a time.

    Args:
        filepath (str): /path/to/file on disk from which json items will be streamed,
            such as items in a JSON array; for example::

                [
                    {"title": "Harrison Bergeron", "text": "The year was 2081, and everybody was finally equal."},
                    {"title": "2BR02B", "text": "Everything was perfectly swell."}
                ]

        mode (str, optional)
        encoding (str, optional)
        prefix (str, optional): if '', the entire JSON object will be read in at once;
            if 'item', each item in a top-level array will be read in successively;
            if 'item.text', each array item's 'text' value will be read in successively

    Yields:
        next matching JSON object; could be a dict, list, int, float, str,
            depending on the value of ``prefix``

    Notes:
        Refer to ``ijson`` at https://pypi.python.org/pypi/ijson/ for usage details.
    """
    with open_sesame(filepath, mode=mode, encoding=encoding) as f:
        if prefix == '':
            yield json.load(f)
        else:
            for item in ijson.items(f, prefix):
                yield item


def read_json_lines(filepath, mode='rt', encoding=None):
    """
    Iterate over a stream of JSON objects, where each line of file ``filepath``
    is a valid JSON object but no JSON object (e.g. array) exists at the top level.

    Args:
        filepath (str): /path/to/file on disk from which json objects will be streamed,
            where each line in the file must be its own json object; for example::

                {"title": "Harrison Bergeron", "text": "The year was 2081, and everybody was finally equal."}\n
                {"title": "2BR02B", "text": "Everything was perfectly swell."}

        mode (str, optional)
        encoding (str, optional)

    Yields:
        dict: next valid JSON object, converted to native Python equivalent
    """
    with open_sesame(filepath, mode=mode, encoding=encoding) as f:
        for line in f:
            yield json.loads(line)


def read_json_mash(filepath, mode='rt', encoding=None, buffersize=2048):
    """
    Iterate over a stream of JSON objects, all of them mashed together, end-to-end,
    on a single line of a file. Bad form, but still manageable.

    Args:
        filepath (str): /path/to/file on disk from which json objects will be streamed,
            where all json objects are mashed together, end-to-end, on a single line,;
            for example::

                {"title": "Harrison Bergeron", "text": "The year was 2081, and everybody was finally equal."}{"title": "2BR02B", "text": "Everything was perfectly swell."}

        mode (str, optional)
        encoding (str, optional)
        buffersize (int, optional): number of bytes to read in as a chunk

    Yields:
        dict: next valid JSON object, converted to native Python equivalent
    """
    with open_sesame(filepath, mode=mode, encoding=encoding) as f:
        buffer = ''
        for chunk in iter(partial(f.read, buffersize), ''):
            buffer += chunk
            while buffer:
                try:
                    result, index = JSON_DECODER.raw_decode(buffer)
                    yield result
                    buffer = buffer[index:]
                # not enough data to decode => read another chunk
                except ValueError:
                    break


def read_csv(filepath, encoding=None, dialect='excel', delimiter=','):
    """
    Iterate over a stream of rows, where each row is an iterable of strings
    and/or numbers with individual values separated by ``delimiter``.

    Args:
        filepath (str): /path/to/file on disk from which rows will be streamed
        encoding (str)
        dialect (str): a grouping of formatting parameters that determine how
            the tabular data is parsed when reading/writing; if 'infer', the
            first 1024 bytes of the file is analyzed, producing a best guess for
            the correct dialect
        delimiter (str): 1-character string used to separate fields in a row

    Yields:
        List[obj]: next row, whose elements are strings and/or numbers

    .. seealso:: https://docs.python.org/3/library/csv.html#csv.reader
    """
    with open_sesame(filepath, mode='rt', encoding=encoding, newline='') as f:
        if dialect == 'infer':
            dialect = csv.Sniffer().sniff(f.read(1024))
            f.seek(0)
        for row in csv.reader(f, dialect=dialect, delimiter=delimiter):
            yield row


def read_spacy_docs(spacy_vocab, filepath):
    """
    Stream ``spacy.Doc`` s from disk at ``filepath`` where they were serialized
    using Spacy's ``spacy.Doc.to_bytes()`` functionality.

    Args:
        spacy_vocab (``spacy.Vocab``): the spacy vocab object used to serialize
            the docs in ``filepath``
        filepath (str): /path/to/file on disk from which spacy docs will be streamed

    Yields:
        the next deserialized ``spacy.Doc``
    """
    with open_sesame(filepath, mode='rb') as f:
        for bytes_string in SpacyDoc.read_bytes(f):
            yield SpacyDoc(spacy_vocab).from_bytes(bytes_string)


def read_sparse_csr_matrix(filepath):
    """
    Read the data, indices, indptr, and shape arrays from a ``.npz`` file on disk
    at ``filepath``, and return an instantiated ``scipy.sparse.csr_matrix``.
    """
    npz_file = np_load(filepath)
    return csr_matrix((npz_file['data'], npz_file['indices'], npz_file['indptr']),
                      shape=npz_file['shape'])


def read_sparse_csc_matrix(filepath):
    """
    Read the data, indices, indptr, and shape arrays from a ``.npz`` file on disk
    at ``filepath``, and return an instantiated ``scipy.sparse.csc_matrix``.
    """
    npz_file = np_load(filepath)
    return csc_matrix((npz_file['data'], npz_file['indices'], npz_file['indptr']),
                      shape=npz_file['shape'])
