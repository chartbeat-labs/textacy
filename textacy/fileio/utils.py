from __future__ import absolute_import, print_function, unicode_literals

import bz2
import gzip
import io
from itertools import tee, starmap
try:
    import lzma
except ImportError:
    pass  # Py2 doesn't include lzma in its stdlib
import os
import zipfile

from cytoolz.itertoolz import cons, pluck

from textacy import compat


def open_sesame(filepath, mode='rt',
                encoding=None, errors=None, newline=None):
    """
    Open file ``filepath``. Compression (if any) is inferred from the file
    extension ('.gz', '.bz2', '.xz', or '.zip') and handled automatically;
    '~', '.', and/or '..' in paths are automatically expanded; if writing to a
    directory that doesn't exist, all intermediate directories are created
    automatically, as needed.

    Args:
        filepath (str): path on disk (absolute or relative) of the file to open
        mode (str): optional string specifying the mode in which ``filepath``
            is opened
        encoding (str): optional name of the encoding used to decode or encode
            ``filepath``; only applicable in text mode
        errors (str): optional string specifying how encoding/decoding errors
            are handled; only applicable in text mode
        newline (str): optional string specifying how universal newlines mode
            works; only applicable in text mode

    Returns:
        file object
    """
    # sanity check args
    if not isinstance(filepath, compat.string_types):
        raise TypeError('filepath must be a string')
    if encoding and 't' not in mode:
        raise ValueError('encoding only applicable for text mode')

    # process filepath and create dirs
    filepath = os.path.realpath(os.path.expanduser(filepath))
    make_dirs(filepath, mode)

    # infer compression from filepath extension
    # and get file handle accordingly
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()
    if ext in ('gz', 'bz2', 'xz', 'zip'):
        # strip bytes/text from mode; 'b' is default, and we'll handle 't' below
        mode_ = mode.replace('b', '').replace('t', '')
        if ext == 'gz':
            f = gzip.GzipFile(filepath, mode=mode_)
        elif ext == 'bz2':
            f = bz2.BZ2File(filepath, mode=mode_)
        elif ext == 'xz':
            if compat.PY2 is True:
                raise IOError("Python2's stdlib doesn't include lzma compression")
            f = lzma.LZMAFile(filepath, mode=mode_)
        elif ext == 'zip':
            f = zipfile.ZipFile(filepath, mode=mode_,
                                compression=zipfile.ZIP_DEFLATED)
            if mode_ == 'r':
                zip_names = f.namelist()
                if len(zip_names) == 0:
                    msg = 'no files found in zip archive "{}"'.format(filepath)
                    raise ValueError(msg)
                elif len(zip_names) == 1:
                    f = f.open(zip_names[0], mode=mode_)
                else:
                    msg = 'multiple files found in zip archive "{}", but only single-file archives supported'.format(filepath)
                    raise ValueError(msg)
        # handle reading/writing compressed files in text mode
        if 't' in mode:
            f = io.TextIOWrapper(f, encoding=encoding, errors=errors, newline=newline)

    # no compression, so file is opened as usual
    else:
        f = io.open(filepath, mode=mode,
                    encoding=encoding, errors=errors, newline=newline)

    return f


def make_dirs(filepath, mode):
    """
    If writing ``filepath`` to a directory that doesn't exist, all intermediate
    directories will be created as needed.
    """
    head, _ = os.path.split(filepath)
    if 'w' in mode and head and not os.path.exists(head):
        os.makedirs(head)


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
        return unzip(((item.pop(content_field), item) for item in items))


def unzip(seq):
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
