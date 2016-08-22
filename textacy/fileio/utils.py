from __future__ import absolute_import, print_function, unicode_literals

import bz2
import gzip
import io
from itertools import tee, starmap
import os
import re
import warnings
try:  # Py3
    import lzma
except ImportError:  # Py2
    pass

from cytoolz.itertoolz import cons, pluck

from textacy import compat


def open_sesame(filepath, mode='rt',
                encoding=None, auto_make_dirs=False,
                errors=None, newline=None):
    """
    Open file ``filepath``. Compression (if any) is inferred from the file
    extension ('.gz', '.bz2', or '.xz') and handled automatically; '~', '.',
    and/or '..' in paths are automatically expanded; if writing to a directory
    that doesn't exist, all intermediate directories can be created
    automatically, as needed.

    `open_sesame` may be used as a drop-in replacement for the built-in `open`.

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
        auto_make_dirs (bool): if True, automatically create (sub)directories if
            not already present in order to write `filepath`

    Returns:
        file object
    """
    # sanity check args
    if not isinstance(filepath, compat.string_types):
        raise TypeError('filepath must be a string, not {}'.format(type(filepath)))
    if encoding and 't' not in mode:
        raise ValueError('encoding only applicable for text mode')

    # process filepath and create dirs
    filepath = os.path.realpath(os.path.expanduser(filepath))
    if auto_make_dirs is True:
        make_dirs(filepath, mode)
    elif 'r' in mode and not os.path.exists(filepath):
        raise OSError('file "{}" does not exist'.format(filepath))

    # infer compression from filepath extension
    # and get file handle accordingly
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()
    if ext in ('.gz', '.bz2', '.xz'):
        # strip bytes/text from mode; 'b' is default, and we'll handle 't' below
        mode_ = mode.replace('b', '').replace('t', '')
        if ext == '.gz':
            f = gzip.GzipFile(filepath, mode=mode_)
        elif ext == '.bz2':
            f = bz2.BZ2File(filepath, mode=mode_)
        elif ext == '.xz':
            if compat.PY2 is True:
                msg = "lzma compression isn't enabled for Python 2; try gzip or bz2"
                raise ValueError(msg)
            f = lzma.LZMAFile(filepath, mode=mode_)
        # handle reading/writing compressed files in text mode
        if 't' in mode:
            if compat.PY2 is True:
                msg = 'Python 2 can\'t read/write compressed files in "{}" mode'.format(mode)
                raise ValueError(msg)
            else:
                f = io.TextIOWrapper(
                    f, encoding=encoding, errors=errors, newline=newline)

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


def coerce_content_type(content, file_mode):
    """
    If the `content` to be written to file and the `file_mode` used to open it
    are incompatible (either bytes with text mode or unicode with bytes mode),
    try to coerce the content type so it can be written.
    """
    if 't' in file_mode and isinstance(content, compat.bytes_type):
        return compat.bytes_to_unicode(content)
    elif 'b' in file_mode and isinstance(content, compat.unicode_type):
        return compat.unicode_to_bytes(content)
    return content


def split_record_fields(items, content_field, itemwise=False):
    """
    Split records' content (text) field from associated metadata fields, but
    keep them paired together for convenient loading into a ``textacy.Doc <textacy.doc.Doc>``
    (with ``itemwise = True``) or ``textacy.Corpus <textacy.corpus.Corpus>``
    (with ``itemwise = False``). Output format depends on the form of the input
    items (dicts vs. lists) and the value for ``itemwise``.

    Args:
        items (Iterable[dict] or Iterable[list]): an iterable of dicts, e.g. as
            read from disk by :func:`read_json_lines() <textacy.fileio.read.read_json_lines>`,
            or an iterable of lists, e.g. as read from disk by :func:`read_csv() <textacy.fileio.read.read_csv>`
        content_field (str or int): if str, key in each dict item whose value is
            the item's content (text); if int, index of the value in each list
            item corresponding to the item's content (text)
        itemwise (bool): if True, content + metadata are paired item-wise
            as an iterable of (content, metadata) 2-tuples; if False, content +
            metadata are paired by position in two parallel iterables in the form of
            a (iterable(content), iterable(metadata)) 2-tuple

    Returns:
        generator(Tuple[str, dict]): if ``itemwise`` is True and ``items`` is an
            iterable of dicts; the first element in each tuple is the item's content,
            the second element is its metadata as a dictionary
        generator(Tuple[str, list]): if ``itemwise`` is True and ``items`` is an
            iterable of lists; the first element in each tuple is the item's content,
            the second element is its metadata as a list
        Tuple[Iterable[str], Iterable[dict]]: if ``itemwise`` is False and ``items``
            is an iterable of dicts; the first element of the tuple is an iterable
            of items' contents, the second is an iterable of their metadata dicts
        Tuple[Iterable[str], Iterable[list]]: if ``itemwise`` is False and ``items``
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
                  match_regex=None, ignore_regex=None,
                  extension=None, ignore_invisible=True, recursive=False):
    """
    Yield full paths of files on disk under directory ``dirname``, optionally
    filtering for or against particular substrings or file extensions and
    crawling all subdirectories.

    Args:
        dirname (str): /path/to/dir on disk where files to read are saved
        match_substr (str): match only files with given substring
            (DEPRECATED; use match_regex)
        ignore_substr (str): match only files *without* given substring
            (DEPRECATED; use ignore_regex)
        match_regex (str): include files whose names match this regex pattern
        ignore_regex (str): include files whose names do *not* match this regex pattern
        extension (str): if files only of a certain type are wanted,
            specify the file extension (e.g. ".txt")
        ignore_invisible (bool): if True, ignore invisible files, i.e.
            those that begin with a period
        recursive (bool): if True, iterate recursively through all files
            in subdirectories; otherwise, only return files directly under
            ``dirname``

    Yields:
        str: next file's name, including the full path on disk

    Raises:
        OSError: if ``dirname`` is not found on disk
    """
    if not os.path.exists(dirname):
        raise OSError('directory "{}" does not exist'.format(dirname))
    # TODO: remove these params in, say, v0.4
    if match_substr or ignore_substr:
        with warnings.catch_warnings():
            warnings.simplefilter('always', DeprecationWarning)
            msg = """
            the `match_substr` and `ignore_substr` params are deprecated!
            use the more flexible `match_regex` and `ignore_regex` params instead
            """.strip().replace('\n', ' ')
            warnings.warn(msg, DeprecationWarning)
    match_regex = re.compile(match_regex) if match_regex else None
    ignore_regex = re.compile(ignore_regex) if ignore_regex else None

    def is_good_file(filename, filepath):
        if ignore_invisible and filename.startswith('.'):
            return False
        if match_substr and match_substr not in filename:
            return False
        if ignore_substr and ignore_substr in filename:
            return False
        if match_regex and not match_regex.search(filename):
            return False
        if ignore_regex and ignore_regex.search(filename):
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
