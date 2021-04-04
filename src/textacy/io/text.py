"""
:mod:`textacy.io.text`: Functions for reading from and writing to disk records in
plain text format, either as one text per file or one text per *line* in a file.
"""
from __future__ import annotations

from typing import Iterable, Optional, Union

from .. import types
from . import utils as io_utils


def read_text(
    filepath: types.PathLike,
    *,
    mode: str = "rt",
    encoding: Optional[str] = None,
    lines: bool = False,
) -> Iterable[str]:
    """
    Read the contents of a text file at ``filepath``, either all at once
    or streaming line-by-line.

    Args:
        filepath: Path to file on disk from which data will be read.
        mode: Mode with which ``filepath`` is opened.
        encoding: Name of the encoding used to decode or encode the data in ``filepath``.
            Only applicable in text mode.
        lines: If False, all data is read in at once;
            otherwise, data is read in one line at a time.

    Yields:
        Next line of text to read in.

        If ``lines`` is False, wrap this output in :func:`next()` to conveniently
        access the full text.
    """
    io_utils._validate_read_mode(mode)
    with io_utils.open_sesame(filepath, mode=mode, encoding=encoding) as f:
        if lines is False:
            yield f.read()
        else:
            for line in f:
                yield line


def write_text(
    data: str | Iterable[str],
    filepath: types.PathLike,
    *,
    mode: str = "wt",
    encoding: Optional[str] = None,
    make_dirs: bool = False,
    lines: bool = False,
) -> None:
    """
    Write text ``data`` to disk at ``filepath``, either all at once
    or streaming line-by-line.

    Args:
        data If ``lines`` is False, a single string to write to disk; for example::

                "isnt rick and morty that thing you get when you die and your body gets all stiff"

            If ``lines`` is True, an iterable of strings to write to disk, one
            item per line; for example::

                ["isnt rick and morty that thing you get when you die and your body gets all stiff",
                 "You're thinking of rigor mortis. Rick and morty is when you get trolled into watching \"never gonna give you up\"",
                 "That's rickrolling. Rick and morty is a type of pasta"]

        filepath: Path to file on disk to which data will be written.
        mode: Mode with which ``filepath`` is opened.
        encoding: Name of the encoding used to decode or encode the data in ``filepath``.
            Only applicable in text mode.
        make_dirs: If True, automatically create (sub)directories
            if not already present in order to write ``filepath``.
        lines: If False, all data is written at once;
            otherwise, data is written to disk one line at a time.
    """
    io_utils._validate_write_mode(mode)
    with io_utils.open_sesame(
        filepath, mode=mode, encoding=encoding, make_dirs=make_dirs
    ) as f:
        if lines is False:
            f.write(data)
        else:
            newline: Union[str, bytes] = "\n" if "t" in mode else b"\n"
            for line in data:
                f.write(line + newline)
