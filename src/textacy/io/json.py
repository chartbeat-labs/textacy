"""
:mod:`textacy.io.json`: Functions for reading from and writing to disk records in JSON format,
as one record per file or one record per *line* in a file.
"""
from __future__ import annotations

import datetime
import functools
import json
from typing import Any, Iterable, Optional, Tuple, Union

from .. import types
from . import utils as io_utils


def read_json(
    filepath: types.PathLike,
    *,
    mode: str = "rt",
    encoding: Optional[str] = None,
    lines: bool = False,
) -> Iterable:
    """
    Read the contents of a JSON file at ``filepath``, either all at once
    or streaming item-by-item.

    Args:
        filepath: Path to file on disk from which data will be read.
        mode: Mode with which ``filepath`` is opened.
        encoding: Name of the encoding used to decode or encode the data
            in ``filepath``. Only applicable in text mode.
        lines: If False, all data is read in at once; otherwise, data is read in
            one line at a time.

    Yields:
        Next JSON item; could be a dict, list, int, float, str,
        depending on the data and the value of ``lines``.
    """
    io_utils._validate_read_mode(mode)
    with io_utils.open_sesame(filepath, mode=mode, encoding=encoding) as f:
        if lines is False:
            yield json.load(f)
        else:
            for line in f:
                yield json.loads(line)


def read_json_mash(
    filepath: types.PathLike,
    *,
    mode: str = "rt",
    encoding: Optional[str] = None,
    buffer_size: int = 2048,
) -> Iterable:
    """
    Read the contents of a JSON file at ``filepath`` one item at a time,
    where all of the items have been mashed together, end-to-end, on a single line.

    Args:
        filepath: Path to file on disk to which data will be written.
        mode: Mode with which ``filepath`` is opened.
        encoding: Name of the encoding used to decode or encode the data
            in ``filepath``. Only applicable in text mode.
        buffer_size: Number of bytes to read in as a chunk.

    Yields:
        Next valid JSON object, converted to native Python equivalent.

    Note:
        Storing JSON data in this format is Not Good. Reading it is doable, so
        this function is included for users' convenience, but note that
        there is no analogous ``write_json_mash()`` function. Don't do it.
    """
    io_utils._validate_read_mode(mode)
    json_decoder = json.JSONDecoder()
    with io_utils.open_sesame(filepath, mode=mode, encoding=encoding) as f:
        buffer_ = ""
        for chunk in iter(functools.partial(f.read, buffer_size), ""):
            buffer_ += chunk
            while buffer_:
                try:
                    result, index = json_decoder.raw_decode(buffer_)
                    yield result
                    buffer_ = buffer_[index:]
                # not enough data to decode => read another chunk
                except ValueError:
                    break


def write_json(
    data: Any,
    filepath: types.PathLike,
    *,
    mode: str = "wt",
    encoding: Optional[str] = None,
    make_dirs: bool = False,
    lines: bool = False,
    ensure_ascii: bool = False,
    separators: Tuple[str, str] = (",", ":"),
    sort_keys: bool = False,
    indent: Optional[int | str] = None,
) -> None:
    """
    Write JSON ``data`` to disk at ``filepath``, either all at once
    or streaming item-by-item.

    Args:
        data: JSON data to write to disk, including any Python objects
            encodable by default in :mod:`json`, as well as dates and datetimes.
            For example::

                [
                    {"title": "Harrison Bergeron", "text": "The year was 2081, and everybody was finally equal."},
                    {"title": "2BR02B", "text": "Everything was perfectly swell."},
                    {"title": "Slaughterhouse-Five", "text": "All this happened, more or less."},
                ]

            If ``lines`` is False, all of ``data`` is written as a single object;
            if True, each item is written to a separate line in ``filepath``.

        filepath: Path to file on disk to which data will be written.
        mode: Mode with which ``filepath`` is opened.
        encoding: Name of the encoding used to decode or encode the data
            in ``filepath``. Only applicable in text mode.
        make_dirs: If True, automatically create (sub)directories if
            not already present in order to write ``filepath``.
        lines: If False, all data is written at once;
            otherwise, data is written to disk one item at a time.
        ensure_ascii: If True, all non-ASCII characters are escaped;
            otherwise, non-ASCII characters are output as-is.
        separators: An (item_separator, key_separator) pair
            specifying how items and keys are separated in output.
        sort_keys: If True, each output dictionary is sorted by key;
            otherwise, dictionary ordering is taken as-is.
        indent: If a non-negative integer or string, items are pretty-printed
            with the specified indent level; if 0, negative, or "", items are separated
            by newlines; if None, the most compact representation is used
            when storing ``data``.

    See Also:
        https://docs.python.org/3/library/json.html#json.dump
    """
    io_utils._validate_write_mode(mode)
    with io_utils.open_sesame(
        filepath, mode=mode, encoding=encoding, make_dirs=make_dirs
    ) as f:
        if lines is False:
            f.write(
                json.dumps(
                    data,
                    indent=indent,
                    ensure_ascii=ensure_ascii,
                    separators=separators,
                    sort_keys=sort_keys,
                    cls=ExtendedJSONEncoder,
                )
            )
        else:
            newline: Union[str, bytes] = "\n" if "t" in mode else b"\n"
            for item in data:
                f.write(
                    json.dumps(
                        item,
                        indent=indent,
                        ensure_ascii=ensure_ascii,
                        separators=separators,
                        sort_keys=sort_keys,
                        cls=ExtendedJSONEncoder,
                    )
                    + newline
                )


class ExtendedJSONEncoder(json.JSONEncoder):
    """
    Sub-class of :class:`json.JSONEncoder`, used to write JSON data to disk in
    :func:`write_json()` while handling a broader range of Python objects.

    - :class:`datetime.datetime` => ISO-formatted string
    - :class:`datetime.date` => ISO-formatted string
    """

    def default(self, obj):
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        else:
            return super().default(obj)
