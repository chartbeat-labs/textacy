"""
JSON
----

Functions for reading from and writing to disk records in JSON format,
as one record per file or one record per *line* in a file.
"""
from __future__ import absolute_import, print_function, unicode_literals

import datetime
import functools
import json

import ijson

from .. import compat
from .utils import open_sesame, _validate_read_mode, _validate_write_mode


def read_json(fname, mode="rt", encoding=None, lines=False):
    """
    Read the contents of a JSON file at ``fname``, either all at once
    or streaming item-by-item.

    Args:
        fname (str): Path to file on disk from which data will be read.
        mode (str): Mode with which ``fname`` is opened.
        encoding (str): Name of the encoding used to decode or encode the data
            in ``fname``. Only applicable in text mode.
        lines (str or bool): If False or '', all data is read in at once; if True,
            items are read in one line at a time; if 'item', each item in the
            top-level array is read in one at a time. Individual fields can also
            be accessed; for example, if 'item.text', each array item's 'text'
            value is read in one at a time.

    Yields:
        object: Next matching JSON item; could be a dict, list, int, float, str,
        depending on the value of ``lines``.

    See Also:
        Refer to the ``ijson`` docs at https://pypi.python.org/pypi/ijson/
        for usage details on specifying (sub-)items via ``lines`` as a string.
    """
    _validate_read_mode(mode)
    if not isinstance(lines, (compat.unicode_, bytes, bool)):
        raise ValueError(
            'lines="{}" is invalid; must be a string or boolean value'.format(lines)
        )
    with open_sesame(fname, mode=mode, encoding=encoding) as f:
        if not lines:
            yield json.load(f)
        elif lines is True:
            for line in f:
                yield json.loads(line)
        elif isinstance(lines, compat.string_types):
            for item in ijson.items(f, lines):
                yield item


def read_json_mash(fname, mode="rt", encoding=None, buffer_size=2048):
    """
    Read the contents of a JSON file at ``fname`` one item at a time,
    where all of the items have been mashed together, end-to-end, on a single line.

    Args:
        fname (str): Path to file on disk to which data will be written.
        mode (str): Mode with which ``fname`` is opened.
        encoding (str): Name of the encoding used to decode or encode the data
            in ``fname``. Only applicable in text mode.
        buffer_size (int): Number of bytes to read in as a chunk.

    Yields:
        object: Next valid JSON object, converted to native Python equivalent.

    Note:
        Storing JSON data in this format is Not Good. Reading it is doable, so
        this function is included for users' convenience, but note that
        there is no analogous ``write_json_mash()`` function. Don't do it.
    """
    _validate_read_mode(mode)
    json_decoder = json.JSONDecoder()
    with open_sesame(fname, mode=mode, encoding=encoding) as f:
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
    data,
    fname,
    mode="wt",
    encoding=None,
    make_dirs=False,
    lines=False,
    ensure_ascii=False,
    separators=(",", ":"),
    sort_keys=False,
    indent=None,
):
    """
    Write JSON ``data`` to disk at ``fname``, either all at once
    or streaming item-by-item.

    Args:
        data (JSON): JSON data to write to disk, including any Python objects
            encodable by default in :mod:`json`, as well as dates and datetimes.
            For example::

                [
                    {"title": "Harrison Bergeron", "text": "The year was 2081, and everybody was finally equal."},
                    {"title": "2BR02B", "text": "Everything was perfectly swell."},
                    {"title": "Slaughterhouse-Five", "text": "All this happened, more or less."},
                ]

            If ``lines`` is False, all of ``data`` is written as a single object;
            if True, each item is written to a separate line in ``fname``.

        fname (str): Path to file on disk to which data will be written.
        mode (str): Mode with which ``fname`` is opened.
        encoding (str): Name of the encoding used to decode or encode the data
            in ``fname``. Only applicable in text mode.
        make_dirs (bool): If True, automatically create (sub)directories if
            not already present in order to write ``fname``.
        lines (bool): If False, all data is written at once; otherwise, data is
            written to disk one item at a time.
        ensure_ascii (bool): If True, all non-ASCII characters are escaped;
            otherwise, non-ASCII characters are output as-is.
        separators (Tuple[str, str]): An (item_separator, key_separator) pair
            specifying how items and keys are separated in output.
        sort_keys (bool): If True, each output dictionary is sorted by key;
            otherwise, dictionary ordering is taken as-is.
        indent (int or str): If a non-negative integer or string, items are
            pretty-printed with the specified indent level; if 0, negative, or "",
            items are separated by newlines; if None, the most compact representation
            is used when storing ``data``.

    See Also:
        https://docs.python.org/3/library/json.html#json.dump
    """
    _validate_write_mode(mode)
    with open_sesame(fname, mode=mode, encoding=encoding, make_dirs=make_dirs) as f:
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
            newline = "\n" if "t" in mode else b"\n"
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
            return super(ExtendedJSONEncoder, self).default(ob)
