"""
Utils
-----

Functions to help read and write data to disk in a variety of formats.
"""
from __future__ import absolute_import, print_function, unicode_literals

import bz2
import gzip
import io
import itertools
import os
import re
import warnings
import zipfile

try:  # Py3
    import lzma
except ImportError:  # Py2
    try:
        from backports import lzma
    except ImportError:  # Py2 without backport installed
        pass

from cytoolz import itertoolz

from .. import compat

_ext_to_compression = {".bz2": "bz2", ".gz": "gzip", ".xz": "xz", ".zip": "zip"}


def open_sesame(
    filepath,
    mode="rt",
    encoding=None,
    errors=None,
    newline=None,
    compression="infer",
    make_dirs=False,
):
    """
    Open file ``filepath``. Automatically handle file compression, relative
    paths and symlinks, and missing intermediate directory creation, as needed.

    ``open_sesame`` may be used as a drop-in replacement for :func:`io.open`.

    Args:
        filepath (str): Path on disk (absolute or relative) of the file to open.
        mode (str): The mode in which ``filepath`` is opened.
        encoding (str): Name of the encoding used to decode or encode ``filepath``.
            Only applicable in text mode.
        errors (str): String specifying how encoding/decoding errors are handled.
            Only applicable in text mode.
        newline (str): String specifying how universal newlines mode works.
            Only applicable in text mode.
        compression (str): Type of compression, if any, with which ``filepath``
            is read from or written to disk. If None, no compression is used;
            if 'infer', compression is inferrred from the extension on ``filepath``.
        make_dirs (bool): If True, automatically create (sub)directories if
            not already present in order to write ``filepath``.

    Returns:
        file object

    Raises:
        TypeError: if ``filepath`` is not a string
        ValueError: if ``encoding`` is specified but ``mode`` is binary
        OSError: if ``filepath`` doesn't exist but ``mode`` is read
    """
    # check args
    if not isinstance(filepath, compat.string_types):
        raise TypeError("filepath must be a string, not {}".format(type(filepath)))
    if encoding and "t" not in mode:
        raise ValueError("encoding only applicable for text mode")

    # normalize filepath and make dirs, as needed
    filepath = os.path.realpath(os.path.expanduser(filepath))
    if make_dirs is True:
        _make_dirs(filepath, mode)
    elif mode.startswith("r") and not os.path.exists(filepath):
        raise OSError('file "{}" does not exist'.format(filepath))

    compression = _get_compression(filepath, compression)

    f = _get_file_handle(
        filepath,
        mode,
        compression=compression,
        encoding=encoding,
        errors=errors,
        newline=newline,
    )

    return f


def _get_compression(filepath, compression):
    """
    Get the compression method for ``filepath``, depending on its file extension
    and the value of ``compression``. Also validate the given values.
    """
    # user has specified "no compression"
    if compression is None:
        return None
    # user wants us to infer compression from filepath
    elif compression == "infer":
        _, ext = os.path.splitext(filepath)
        try:
            return _ext_to_compression[ext.lower()]
        except KeyError:
            return None
    # user has specified compression; validate it
    elif compression in _ext_to_compression.values():
        return compression
    else:
        raise ValueError(
            'compression="{}" is invalid; '
            "valid values are {}.".format(
                compression, [None, "infer"] + sorted(_ext_to_compression.values())
            )
        )


def _get_file_handle(
    filepath, mode, compression=None, encoding=None, errors=None, newline=None
):
    """
    Get a file handle for the given ``filepath`` and ``mode``, plus optional kwargs.
    """
    if compression:

        mode_ = mode.replace("b", "").replace("t", "")

        if compression == "gzip":
            f = gzip.GzipFile(filepath, mode=mode_)
        elif compression == "bz2":
            f = bz2.BZ2File(filepath, mode=mode_)
        elif compression == "xz":
            try:
                f = lzma.LZMAFile(filepath, mode=mode_)
            except NameError:
                raise ValueError(
                    "lzma compression isn't included in Python 2's stdlib; "
                    "try gzip or bz2, or install `backports.lzma`"
                )
        elif compression == "zip":
            zip_file = zipfile.ZipFile(filepath, mode=mode_)
            zip_names = zip_file.namelist()
            if len(zip_names) == 1:
                f = zip_file.open(zip_names[0])
            elif len(zip_names) == 0:
                raise ValueError('no files found in zip file "{}"'.format(filepath))
            else:
                raise ValueError(
                    '{} files found in zip file "{}", '
                    "but only one file is allowed.".format(len(zip_names), filepath)
                )
        else:
            raise ValueError(
                'compression="{}" is invalid; '
                "valid values are {}.".format(
                    compression, [None, "infer"] + sorted(_ext_to_compression.values())
                )
            )

        if "t" in mode:
            if compat.is_python2 is True:
                raise ValueError(
                    'Python 2 can\'t open compressed files in "{}" mode'.format(mode)
                )
            else:
                f = io.TextIOWrapper(
                    f, encoding=encoding, errors=errors, newline=newline
                )

    # no compression, file is opened as usual
    else:
        f = io.open(
            filepath, mode=mode, encoding=encoding, errors=errors, newline=newline
        )

    return f


def _make_dirs(filepath, mode):
    """
    If writing ``filepath`` to a directory that doesn't exist, all intermediate
    directories will be created as needed.
    """
    head, _ = os.path.split(filepath)
    if "w" in mode and head and not os.path.exists(head):
        os.makedirs(head)


def _validate_read_mode(mode):
    if "w" in mode or "a" in mode:
        raise ValueError(
            'mode="{}" is invalid; file must be opened in read mode'.format(mode)
        )


def _validate_write_mode(mode):
    if "r" in mode:
        raise ValueError(
            'mode="{}" is invalid; file must be opened in write mode'.format(mode)
        )


def coerce_content_type(content, file_mode):
    """
    If the `content` to be written to file and the `file_mode` used to open it
    are incompatible (either bytes with text mode or unicode with bytes mode),
    try to coerce the content type so it can be written.
    """
    if "t" in file_mode and isinstance(content, compat.bytes_):
        return compat.bytes_to_unicode(content)
    elif "b" in file_mode and isinstance(content, compat.unicode_):
        return compat.unicode_to_bytes(content)
    return content


def split_records(items, content_field, itemwise=False):
    """
    Split records' content (text) from associated metadata, but keep them paired
    together for convenient loading into a :class:`textacy.Doc <textacy.doc.Doc>`
    (with ``itemwise=True``) or :class:`textacy.Corpus <textacy.corpus.Corpus>`
    (with ``itemwise=False``).

    Args:
        items (Iterable[dict] or Iterable[list]): An iterable of dicts, e.g. as
            read from disk by :func:`read_json(lines=True) <textacy.io.json.read_json>`,
            or an iterable of lists, e.g. as read from disk by
            :func:`read_csv() <textacy.io.csv.read_csv>`.
        content_field (str or int): If str, key in each dict item whose value is
            the item's content (text); if int, index of the value in each list
            item corresponding to the item's content (text).
        itemwise (bool): If True, content + metadata are paired item-wise
            as an iterable of (content, metadata) 2-tuples; if False, content +
            metadata are paired by position in two parallel iterables in the form of
            a (iterable(content), iterable(metadata)) 2-tuple.

    Returns:
        Generator(Tuple[str, dict]): If ``itemwise`` is True and ``items`` is Iterable[dict];
        the first element in each tuple is the item's content,
        the second element is its metadata as a dictionary.

        Generator(Tuple[str, list]): If ``itemwise`` is True and ``items`` is Iterable[list];
        the first element in each tuple is the item's content,
        the second element is its metadata as a list.

        Tuple[Iterable[str], Iterable[dict]]: If ``itemwise`` is False and
        ``items`` is Iterable[dict];
        the first element of the tuple is an iterable of items' contents,
        the second is an iterable of their metadata dicts.

        Tuple[Iterable[str], Iterable[list]]: If ``itemwise`` is False and
        ``items`` is Iterable[list];
        the first element of the tuple is an iterable of items' contents,
        the second is an iterable of their metadata lists.
    """
    if itemwise is True:
        return ((item.pop(content_field), item) for item in items)
    else:
        return unzip(((item.pop(content_field), item) for item in items))


def split_record_fields(items, content_field, itemwise=False):
    """
    This functionality has been moved to :func:`split_records()`, and this is just
    a temporary alias for that other function. You should use it instead of this.
    """
    warnings.warn(
        "`split_record_fields()` has been renamed `split_records()`, "
        "and this function is just a temporary alias for it.",
        DeprecationWarning,
    )
    return split_records(items, content_field, itemwise=False)


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
    seqs = itertools.tee(itertoolz.cons(first, seq), niters)
    return tuple(itertools.starmap(itertoolz.pluck, enumerate(seqs)))


def get_filenames(
    dirname,
    match_regex=None,
    ignore_regex=None,
    extension=None,
    ignore_invisible=True,
    recursive=False,
):
    """
    Yield full paths of files on disk under directory ``dirname``, optionally
    filtering for or against particular patterns or file extensions and
    crawling all subdirectories.

    Args:
        dirname (str): Path to directory on disk where files are stored.
        match_regex (str): Regular expression pattern. Only files whose names
            match this pattern are included.
        ignore_regex (str): Regular expression pattern. Only files whose names
            *do not* match this pattern are included.
        extension (str): File extension, e.g. ".txt" or ".json". Only files
            whose extensions match are included.
        ignore_invisible (bool): If True, ignore invisible files, i.e.
            those that begin with a period.; otherwise, include them.
        recursive (bool): If True, iterate recursively through subdirectories
            in search of files to include; otherwise, only return files located
            directly under ``dirname``.

    Yields:
        str: Next file's name, including the full path on disk.

    Raises:
        OSError: if ``dirname`` is not found on disk
    """
    if not os.path.exists(dirname):
        raise OSError('directory "{}" does not exist'.format(dirname))
    match_regex = re.compile(match_regex) if match_regex else None
    ignore_regex = re.compile(ignore_regex) if ignore_regex else None

    def is_good_file(filename, filepath):
        if ignore_invisible and filename.startswith("."):
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
            if ignore_invisible and dirpath.startswith("."):
                continue
            for filename in filenames:
                if filename.startswith("."):
                    continue
                if is_good_file(filename, dirpath):
                    yield os.path.join(dirpath, filename)
    else:
        for filename in os.listdir(dirname):
            if is_good_file(filename, dirname):
                yield os.path.join(dirname, filename)
