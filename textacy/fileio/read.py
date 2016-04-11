"""
Module with functions for reading content from disk in common formats.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from functools import partial
import gzip
import io
from itertools import tee, starmap
import json
import os

from cytoolz.itertoolz import cons, pluck
import ijson
from numpy import load as np_load
from scipy.sparse import csc_matrix, csr_matrix
from spacy.tokens.doc import Doc as SpacyDoc

from textacy.compat import bzip_open

JSON_DECODER = json.JSONDecoder()


def read_json(filename, mode='rt', encoding=None, prefix=''):
    """
    Iterate over JSON objects matching the field given by ``prefix``.
    Useful for reading a large JSON array one item (with ``prefix='item'``)
    or sub-item (``prefix='item.fieldname'``) at a time.

    Args:
        filename (str): /path/to/file on disk from which json items will be streamed,
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
    with io.open(filename, mode=mode, encoding=encoding) as f:
        if prefix == '':
            yield json.load(f)
        else:
            for item in ijson.items(f, prefix):
                yield item


def read_json_lines(filename, mode='rt', encoding=None):
    """
    Iterate over a stream of JSON objects, where each line of file ``filename``
    is a valid JSON object but no JSON object (e.g. array) exists at the top level.

    Args:
        filename (str): /path/to/file on disk from which json objects will be streamed,
            where each line in the file must be its own json object; for example::

                {"title": "Harrison Bergeron", "text": "The year was 2081, and everybody was finally equal."}\n
                {"title": "2BR02B", "text": "Everything was perfectly swell."}

        mode (str, optional)
        encoding (str, optional)

    Yields:
        dict: next valid JSON object, converted to native Python equivalent
    """
    with io.open(filename, mode=mode, encoding=encoding) as f:
        for line in f:
            yield json.loads(line)


def read_json_mash(filename, mode='rt', encoding=None, buffersize=2048):
    """
    Iterate over a stream of JSON objects, all of them mashed together, end-to-end,
    on a single line of a file. Bad form, but still manageable.

    Args:
        filename (str): /path/to/file on disk from which json objects will be streamed,
            where all json objects are mashed together, end-to-end, on a single line,;
            for example::

                {"title": "Harrison Bergeron", "text": "The year was 2081, and everybody was finally equal."}{"title": "2BR02B", "text": "Everything was perfectly swell."}

        mode (str, optional)
        encoding (str, optional)
        buffersize (int, optional): number of bytes to read in as a chunk

    Yields:
        dict: next valid JSON object, converted to native Python equivalent
    """
    with io.open(filename, mode=mode, encoding=encoding) as f:
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


def split_content_and_metadata(items, content_field, itemwise=True):
    """
    Split content (text) from associated metadata, but keep them paired together,
    for convenient loading into a ``TextDoc`` (with ``itemwise = True``) or
    ``TextCorpus.from_texts()`` (with ``itemwise = False``). Output format depends
    on the form of the input items (dicts vs. lists) and the value for ``itemwise``.

    Args:
        items (iterable(dict) or iterable(list)): an iterable of dicts, e.g. as
            read from disk by :func:`read_json_lines() <textacy.fileio.read.read_json_lines>`,
            or an iterable of lists, e.g. as streamed from a Wikipedia database dump::

                >>> ([pageid, title, text] for pageid, title, text in
                ...  textacy.corpora.wikipedia.get_plaintext_pages(<WIKIFILE>))

        content_field (str or int): if str, key in each dict item whose value is
            the item's content (text); if int, index of the value in each list item
            corresponding to the item's content (text)
        itemwise (bool, optional): if True, content + metadata are paired item-wise
            as an iterable of (content, metadata) 2-tuples; if False, content +
            metadata are paired by position in two parallel iterables in the form of
            a (iterable(content), iterable(metadata)) 2-tuple

    Returns:
        generator(tuple(str, dict)): if ``itemwise`` is True and ``items`` is an
            iterable of dicts; the first element in each tuple is the item's content,
            the second element is its metadata as a dictionary
        generator(tuple(str, list)): if ``itemwise`` is True and ``items`` is an
            iterable of lists; the first element in each tuple is the item's content,
            the second element is its metadata as a list
        tuple(iterable(str), iterable(dict)): if ``itemwise`` is False and ``items``
            is an iterable of dicts; the first element of the tuple is an iterable
            of items' contents, the second is an iterable of their metadata dicts
        tuple(iterable(str), iterable(list)): if ``itemwise`` is False and ``items``
            is an iterable of lists; the first element of the tuple is an iterable
            of items' contents, the second is an iterable of their metadata lists
    """
    if itemwise is True:
        return ((item.pop(content_field), item) for item in items)
    else:
        return _unzip(((item.pop(content_field), item) for item in items))


def _unzip(seq):
    """
    Borrowed from ``toolz.sandbox.core.unzip``, but using cytoolz instead of toolz
    to avoid the additional dependency.
    """
    seq = iter(seq)
    # check how many iterators we need
    try:
        first = tuple(next(seq))
    except StopIteration:
        return tuple()
    # and create them
    niters = len(first)
    seqs = tee(cons(first, seq), niters)
    return tuple(starmap(pluck, enumerate(seqs)))


def read_file(filename, mode='rt', encoding=None):
    """
    Read the full contents of a file. Files compressed with gzip or bz2 are handled
    automatically.
    """
    _open = gzip.open if filename.endswith('.gz') \
        else bzip_open if filename.endswith('.bz2') \
        else io.open
    try:
        with _open(filename, mode=mode, encoding=encoding) as f:
            return f.read()
    except TypeError:  # Py2's bz2.BZ2File doesn't accept `encoding` ...
        with _open(filename, mode=mode) as f:
            return f.read()


def read_file_lines(filename, mode='rt', encoding=None):
    """
    Read the contents of a file, line by line. Files compressed with gzip or bz2
    are handled automatically.
    """
    _open = gzip.open if filename.endswith('.gz') \
        else bzip_open if filename.endswith('.bz2') \
        else io.open
    try:
        with _open(filename, mode=mode, encoding=encoding) as f:
            for line in f:
                yield line
    except TypeError:  # Py2's bz2.BZ2File doesn't accept `encoding` ...
        with _open(filename, mode=mode) as f:
            for line in f:
                yield line


def read_spacy_docs(spacy_vocab, filename):
    """
    Stream ``spacy.Doc`` s from disk at ``filename`` where they were serialized
    using Spacy's ``spacy.Doc.to_bytes()`` functionality.

    Args:
        spacy_vocab (``spacy.Vocab``): the spacy vocab object used to serialize
            the docs in ``filename``
        filename (str): /path/to/file on disk from which spacy docs will be streamed

    Yields:
        the next deserialized ``spacy.Doc``
    """
    with io.open(filename, mode='rb') as f:
        for bytes_string in SpacyDoc.read_bytes(f):
            yield SpacyDoc(spacy_vocab).from_bytes(bytes_string)


def read_sparse_csr_matrix(filename):
    """
    Read the data, indices, indptr, and shape arrays from a ``.npz`` file on disk
    at ``filename``, and return an instantiated ``scipy.sparse.csr_matrix``.
    """
    npz_file = np_load(filename)
    return csr_matrix((npz_file['data'], npz_file['indices'], npz_file['indptr']),
                      shape=npz_file['shape'])


def read_sparse_csc_matrix(filename):
    """
    Read the data, indices, indptr, and shape arrays from a ``.npz`` file on disk
    at ``filename``, and return an instantiated ``scipy.sparse.csc_matrix``.
    """
    npz_file = np_load(filename)
    return csc_matrix((npz_file['data'], npz_file['indices'], npz_file['indptr']),
                      shape=npz_file['shape'])


def get_filenames(dirname, match_substr=None, ignore_substr=None,
                  extension=None, ignore_invisible=True, recursive=False):
    """
    Yield the full paths of files on disk under directory ``dirname``, optionally
    filtering for or against particular substrings or file extensions. and crawling all subdirectories.

    Args:
        dirname (str): /path/to/dir on disk where files to read are saved
        match_substr (str, optional): match only files with given substring
        ignore_substr (str, optional): match only files *without* given substring
        extension (str, optional): if files only of a certain type are wanted,
            specify the file extension (e.g. ".txt")
        ignore_invisible (bool, optional): if True, ignore invisible files, i.e.
            those that begin with a period
        recursive (bool, optional): if True, iterate recursively through all files
            in subdirectories; otherwise, only return files directly under ``dirname``

    Yields:
        str: next file's name, including the full path on disk

    Raises:
        IOError: if ``dirname`` is not found on disk
    """
    if not os.path.exists(dirname):
        raise IOError('directory {} does not exist'.format(dirname))

    def is_good_file(filename, filepath):
        if ignore_invisible and filename.startswith('.'):
            return False
        if match_substr and match_substr not in filename:
            return False
        if ignore_substr and ignore_substr in filename:
            return False
        if extension and not os.path.splitext(filename)[-1] == extension:
            return False
        if not os.path.isfile(os.path.join(filepath, filename)):
            return False
        return True

    if recursive is True:
        for dirpath, _, filenames in os.walk(dirname):
            if ignore_invisible and dirpath.startswith('.'):
                continue
            for filename in filenames:
                if filename.startswith('.'):
                    continue
                if is_good_file(filename, dirpath):
                    yield os.path.join(dirpath, filename)
    else:
        for filename in os.listdir(dirname):
            if is_good_file(filename, dirname):
                yield os.path.join(dirname, filename)
