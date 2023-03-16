"""
I/O Utils
---------

:mod:`textacy.io.utils`: Functions to help read and write data to disk
in a variety of formats.
"""
from __future__ import annotations

import bz2
import gzip
import io
import itertools
import logging
import lzma
import os
import re
import shutil
import tarfile
import urllib
import zipfile
from typing import IO, Iterable, Literal, Optional

from cytoolz import itertoolz

from .. import constants
from .. import errors as errors_
from .. import types, utils
from .http import write_http_stream


LOGGER = logging.getLogger(__name__)

_ext_to_compression = {".bz2": "bz2", ".gz": "gzip", ".xz": "xz", ".zip": "zip"}


def open_sesame(
    filepath: types.PathLike,
    *,
    mode: str = "rt",
    encoding: Optional[str] = None,
    errors: Optional[str] = None,
    newline: Optional[str] = None,
    compression: Literal["infer", "bz2", "gzip", "xz", "zip"] = "infer",
    make_dirs: bool = False,
) -> IO:
    """
    Open file ``filepath``. Automatically handle file compression, relative
    paths and symlinks, and missing intermediate directory creation, as needed.

    ``open_sesame`` may be used as a drop-in replacement for :func:`io.open`.

    Args:
        filepath: Path on disk (absolute or relative) of the file to open.
        mode: The mode in which ``filepath`` is opened.
        encoding: Name of the encoding used to decode or encode ``filepath``.
            Only applicable in text mode.
        errors: String specifying how encoding/decoding errors are handled.
            Only applicable in text mode.
        newline: String specifying how universal newlines mode works.
            Only applicable in text mode.
        compression: Type of compression, if any, with which ``filepath``
            is read from or written to disk. If None, no compression is used;
            if 'infer', compression is inferred from the extension on ``filepath``.
        make_dirs: If True, automatically create (sub)directories
            if not already present in order to write ``filepath``.

    Returns:
        file object

    Raises:
        TypeError: if ``filepath`` is not a string
        ValueError: if ``encoding`` is specified but ``mode`` is binary
        OSError: if ``filepath`` doesn't exist but ``mode`` is read
    """
    # check args
    if encoding and "t" not in mode:
        raise ValueError("encoding only applicable for text mode")

    # normalize filepath and make dirs, as needed
    filepath = utils.to_path(filepath).resolve()
    if make_dirs is True:
        _make_dirs(filepath, mode)
    elif mode.startswith("r") and not filepath.is_file():
        raise OSError(f"file '{filepath}' does not exist")

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
        ext = filepath.suffix
        try:
            return _ext_to_compression[ext.lower()]
        except KeyError:
            return None
    # user has specified compression; validate it
    elif compression in _ext_to_compression.values():
        return compression
    else:
        valid_values = [None, "infer"] + sorted(_ext_to_compression.values())
        raise ValueError(
            errors_.value_invalid_msg("compression", compression, valid_values)
        )


def _get_file_handle(
    filepath,
    mode,
    *,
    compression=None,
    encoding=None,
    errors=None,
    newline=None,
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
            f = lzma.LZMAFile(filepath, mode=mode_)
        elif compression == "zip":
            zip_file = zipfile.ZipFile(filepath, mode=mode_)
            zip_names = zip_file.namelist()
            if len(zip_names) == 1:
                f = zip_file.open(zip_names[0])
            elif len(zip_names) == 0:
                raise ValueError(f"no files found in zip file '{filepath}'")
            else:
                raise ValueError(
                    f"{len(zip_names)} files found in zip file '{filepath}', "
                    "but only one file is allowed"
                )
        else:
            valid_values = [None, "infer"] + sorted(_ext_to_compression.values())
            raise ValueError(
                errors_.value_invalid_msg("compression", compression, valid_values)
            )

        if "t" in mode:
            f = io.TextIOWrapper(f, encoding=encoding, errors=errors, newline=newline)

    # no compression, file is opened as usual
    else:
        f = filepath.open(mode=mode, encoding=encoding, errors=errors, newline=newline)

    return f


def _make_dirs(filepath, mode):
    """
    If writing ``filepath`` to a directory that doesn't exist, all intermediate
    directories will be created as needed.
    """
    parent = filepath.parent
    if "w" in mode and parent:
        os.makedirs(parent, exist_ok=True)


def _validate_read_mode(mode):
    if "w" in mode or "a" in mode:
        raise ValueError(
            f"mode = '{mode}' is invalid; file must be opened in read mode"
        )


def _validate_write_mode(mode):
    if "r" in mode:
        raise ValueError(
            f"mode = '{mode}' is invalid; file must be opened in write mode"
        )


def coerce_content_type(content: types.AnyStr, file_mode: str) -> str | bytes:
    """
    If the `content` to be written to file and the `file_mode` used to open it
    are incompatible (either bytes with text mode or unicode with bytes mode),
    try to coerce the content type so it can be written.
    """
    if "t" in file_mode:
        return utils.to_unicode(content)
    elif "b" in file_mode:
        return utils.to_bytes(content)
    return content


def split_records(
    items: Iterable,
    content_field: str | int,
    itemwise: bool = False,
) -> Iterable:
    """
    Split records' content (text) from associated metadata, but keep them paired
    together.

    Args:
        items: An iterable of dicts, e.g. as read from disk by
            :func:`read_json(lines=True) <textacy.io.json.read_json>`,
            or an iterable of lists, e.g. as read from disk by
            :func:`read_csv() <textacy.io.csv.read_csv>`.
        content_field: If str, key in each dict item whose value is the item's content (text);
            if int, index of the value in each list item corresponding to the item's content (text).
        itemwise: If True, content + metadata are paired item-wise
            as an iterable of (content, metadata) 2-tuples; if False, content +
            metadata are paired by position in two parallel iterables in the form of
            a (iterable(content), iterable(metadata)) 2-tuple.

    Returns:
        Generator(tuple[str, dict]): If ``itemwise`` is True and ``items`` is Iterable[dict];
        the first element in each tuple is the item's content,
        the second element is its metadata as a dictionary.

        Generator(tuple[str, list]): If ``itemwise`` is True and ``items`` is Iterable[list];
        the first element in each tuple is the item's content,
        the second element is its metadata as a list.

        tuple[Iterable[str], Iterable[dict]]: If ``itemwise`` is False and
        ``items`` is Iterable[dict];
        the first element of the tuple is an iterable of items' contents,
        the second is an iterable of their metadata dicts.

        tuple[Iterable[str], Iterable[list]]: If ``itemwise`` is False and
        ``items`` is Iterable[list];
        the first element of the tuple is an iterable of items' contents,
        the second is an iterable of their metadata lists.
    """
    if itemwise is True:
        return ((item.pop(content_field), item) for item in items)
    else:
        return unzip(((item.pop(content_field), item) for item in items))


def unzip(seq: Iterable) -> tuple:
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


def get_filepaths(
    dirpath: types.PathLike,
    *,
    match_regex: Optional[str] = None,
    ignore_regex: Optional[str] = None,
    extension: Optional[str] = None,
    ignore_invisible: bool = True,
    recursive: bool = False,
) -> Iterable[str]:
    """
    Yield full paths of files on disk under directory ``dirpath``, optionally
    filtering for or against particular patterns or file extensions and
    crawling all subdirectories.

    Args:
        dirpath: Path to directory on disk where files are stored.
        match_regex: Regular expression pattern.
            Only files whose names match this pattern are included.
        ignore_regex: Regular expression pattern.
            Only files whose names *do not* match this pattern are included.
        extension: File extension, e.g. ".txt" or ".json".
            Only files whose extensions match are included.
        ignore_invisible: If True, ignore invisible files,
            i.e. those that begin with a period.; otherwise, include them.
        recursive: If True, iterate recursively through subdirectories
            in search of files to include; otherwise, only return files located
            directly under ``dirpath``.

    Yields:
        Next file's name, including the full path on disk.

    Raises:
        OSError: if ``dirpath`` is not found on disk
    """
    dirpath = utils.to_path(dirpath).resolve()
    if not dirpath.is_dir():
        raise OSError(f"directory '{dirpath}' does not exist")

    re_match = re.compile(match_regex) if match_regex else None
    re_ignore = re.compile(ignore_regex) if ignore_regex else None

    def is_good_file(dpath, fname):
        if ignore_invisible and fname.startswith("."):
            return False
        if re_match and not re_match.search(fname):
            return False
        if re_ignore and re_ignore.search(fname):
            return False
        if extension and os.path.splitext(fname)[-1] != extension:
            return False
        if not os.path.isfile(os.path.join(dpath, fname)):
            return False
        return True

    if recursive is True:
        for dirpath_, _, filenames in os.walk(dirpath):
            if ignore_invisible and dirpath_.startswith("."):
                continue
            for filename in filenames:
                if filename.startswith("."):
                    continue
                if is_good_file(dirpath_, filename):
                    yield os.path.join(dirpath_, filename)
    else:
        for subpath in dirpath.iterdir():
            if is_good_file(str(dirpath), subpath.name):
                yield str(subpath)


def download_file(
    url: str,
    *,
    filename: Optional[str] = None,
    dirpath: types.PathLike = constants.DEFAULT_DATA_DIR,
    force: bool = False,
) -> Optional[str]:
    """
    Download a file from ``url`` and save it to disk.

    Args:
        url: Web address from which to download data.
        filename: Name of the file to which downloaded data is saved.
            If None, a filename will be inferred from the ``url``.
        dirpath: Full path to the directory on disk
            under which downloaded data will be saved as ``filename``.
        force: If True, download the data even if it already exists at
            ``dirpath/filename``; otherwise, only download if the data doesn't
            already exist on disk.

    Returns:
        Full path of file saved to disk.
    """
    if not filename:
        filename = get_filename_from_url(url)
    filepath = utils.to_path(dirpath).resolve() / filename
    if filepath.is_file() and force is False:
        LOGGER.info(
            "file '%s' already exists and force=False; skipping download...",
            filepath,
        )
        return None
    else:
        write_http_stream(url, filepath, mode="wb", make_dirs=True)
    return str(filepath)


def get_filename_from_url(url: str) -> str:
    """
    Derive a filename from a URL's path.

    Args:
        url: URL from which to extract a filename.

    Returns:
        Filename in URL.
    """
    return os.path.basename(urllib.parse.urlparse(urllib.parse.unquote_plus(url)).path)


def unpack_archive(
    filepath: types.PathLike, *, extract_dir: Optional[types.PathLike] = None
) -> types.PathLike:
    """
    Extract data from a zip or tar archive file into a directory
    (or do nothing if the file isn't an archive).

    Args:
        filepath: Full path to file on disk from which archived contents will be extracted.
        extract_dir: Full path of the directory into which contents will be extracted.
            If not provided, the same directory as ``filepath`` is used.

    Returns:
        Path to directory of extracted contents.
    """
    filepath = utils.to_path(filepath).resolve()
    if not extract_dir:
        extract_dir = str(filepath.parent)
    filepath = str(filepath)
    os.makedirs(extract_dir, exist_ok=True)
    is_zipfile = zipfile.is_zipfile(filepath)
    is_tarfile = tarfile.is_tarfile(filepath)
    if not is_zipfile and not is_tarfile:
        LOGGER.debug("'%s' is not an archive", filepath)
        return extract_dir
    else:
        LOGGER.info("extracting data from archive file '%s'", filepath)
        shutil.unpack_archive(filepath, extract_dir=extract_dir, format=None)
        # we want to rename the unpacked directory to a consistent value
        # unfortunately, shutil doesn't pass this back to us
        # so, we get the root path of all the constituent members
        if is_zipfile:
            with zipfile.ZipFile(filepath, mode="r") as zf:
                members = zf.namelist()
        else:
            with tarfile.open(filepath, mode="r") as tf:
                members = tf.getnames()
        src_basename = os.path.commonpath(members)
        dest_basename = os.path.basename(filepath)
        if src_basename:
            while True:
                tmp, _ = os.path.splitext(dest_basename)
                if tmp == dest_basename:
                    break
                else:
                    dest_basename = tmp
            if src_basename != dest_basename:
                return shutil.move(
                    os.path.join(extract_dir, src_basename),
                    os.path.join(extract_dir, dest_basename),
                )
            else:
                return os.path.join(extract_dir, src_basename)
        else:
            return extract_dir
