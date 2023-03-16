"""
:mod:`textacy.io.csv`: Functions for reading from and writing to disk records in CSV format,
where CSVs may be delimited not only by commas (the default) but tabs, pipes, and
other valid one-char delimiters.
"""
from __future__ import annotations

import csv
from typing import Any, Iterable, Iterator, Optional, Sequence, Type, Union

from .. import types
from . import utils as io_utils


def read_csv(
    filepath: types.PathLike,
    *,
    encoding: Optional[str] = None,
    fieldnames: Optional[str | Sequence[str]] = None,
    dialect: str | Type[csv.Dialect] = "excel",
    delimiter: str = ",",
    quoting: int = csv.QUOTE_NONNUMERIC,
) -> Iterable[list] | Iterable[dict]:
    """
    Read the contents of a CSV file at ``filepath``, streaming line-by-line,
    where each line is a list of strings and/or floats whose values
    are separated by ``delimiter``.

    Args:
        filepath: Path to file on disk from which data will be read.
        encoding: Name of the encoding used to decode or encode the data in ``filepath``.
        fieldnames: If specified, gives names for columns of values,
            which are used as keys in an ordered dictionary representation
            of each line's data. If 'infer', the first kB of data is analyzed
            to make a guess about whether the first row is a header of column
            names, and if so, those names are used as keys. If None, no column
            names are used, and each line is returned as a list of strings/floats.
        dialect: Grouping of formatting parameters that determine how
            the data is parsed when reading/writing. If 'infer', the first kB
            of data is analyzed to get a best guess for the correct dialect.
        delimiter: 1-character string used to separate fields in a row.
        quoting: Type of quoting to apply to field values. See:
            https://docs.python.org/3/library/csv.html#csv.QUOTE_NONNUMERIC

    Yields:
        List[obj]: Next row, whose elements are strings and/or floats.
        If ``fieldnames`` is None or 'infer' doesn't detect a header row.

        *or*

        dict[str, obj]: Next row, as an ordered dictionary of (key, value) pairs,
        where keys are column names and values are the corresponding strings
        and/or floats. If ``fieldnames`` is a list of column names or 'infer'
        detects a header row.

    See Also:
        https://docs.python.org/3/library/csv.html#csv.reader
    """
    has_header = False
    with io_utils.open_sesame(filepath, mode="rt", encoding=encoding, newline="") as f:
        if dialect == "infer" or fieldnames == "infer":
            sniffer = csv.Sniffer()
            # add pipes to the list of preferred delimiters, and put spaces last
            sniffer.preferred = [",", "\t", "|", ";", ":", " "]
            # sample = "".join(f.readline() for _ in range(5))  # f.read(1024)
            sample = f.read(1024)
            if dialect == "infer":
                dialect = sniffer.sniff(sample)
            if fieldnames == "infer":
                has_header = sniffer.has_header(sample)
            f.seek(0)
        csv_reader: Union[csv.DictReader, Iterator]
        if has_header is True:
            csv_reader = csv.DictReader(
                f,
                fieldnames=None,
                dialect=dialect,
                delimiter=delimiter,
                quoting=quoting,
            )
        elif fieldnames:
            csv_reader = csv.DictReader(
                f,
                fieldnames=fieldnames,
                dialect=dialect,
                delimiter=delimiter,
                quoting=quoting,
            )
            first_row = next(csv_reader)
            # is the first row a header with same values as fieldnames?
            # if not, we should yield the row as usual
            if not all(key == value for key, value in first_row.items()):
                yield first_row
        else:
            csv_reader = csv.reader(
                f,
                dialect=dialect,
                delimiter=delimiter,
                quoting=quoting,
            )
        for row in csv_reader:
            yield row


def write_csv(
    data: Iterable[dict[str, Any]] | Iterable[Iterable],
    filepath: types.PathLike,
    *,
    encoding: Optional[str] = None,
    make_dirs: bool = False,
    fieldnames: Optional[Sequence[str]] = None,
    dialect: str = "excel",
    delimiter: str = ",",
    quoting: int = csv.QUOTE_NONNUMERIC,
) -> None:
    """
    Write rows of ``data`` to disk at ``filepath``, where each row is an iterable
    or a dictionary of strings and/or numbers, written to one line with values
    separated by ``delimiter``.

    Args:
        data: If ``fieldnames`` is None, an iterable of iterables of strings
            and/or numbers to write to disk; for example::

                [['That was a great movie!', 0.9],
                 ['The movie was okay, I guess.', 0.2],
                 ['Worst. Movie. Ever.', -1.0]]

            If ``fieldnames`` is specified, an iterable of dictionaries with
            string and/or number values to write to disk; for example::

                [{'text': 'That was a great movie!', 'score': 0.9},
                 {'text': 'The movie was okay, I guess.', 'score': 0.2},
                 {'text': 'Worst. Movie. Ever.', 'score': -1.0}]

        filepath: Path to file on disk to which data will be written.
        encoding: Name of the encoding used to decode or encode the data in ``filepath``.
        make_dirs: If True, automatically create (sub)directories if not already present
            in order to write ``filepath``.
        fieldnames: Sequence of keys that identify the order in which
            values in each rows' dictionary is written to ``filepath``. These are
            included in ``filepath`` as a header row of column names.

            .. note:: Only specify this if ``data`` is an iterable of dictionaries.

        dialect: Grouping of formatting parameters that determine how
            the data is parsed when reading/writing.
        delimiter: 1-character string used to separate fields in a row.
        quoting: Type of quoting to apply to field values. See:
            https://docs.python.org/3/library/csv.html#csv.QUOTE_NONNUMERIC

    See Also:
        https://docs.python.org/3/library/csv.html#csv.writer
    """
    with io_utils.open_sesame(
        filepath, mode="wt", newline="", encoding=encoding, make_dirs=make_dirs
    ) as f:
        csv_writer: Union[csv.DictWriter, Any]
        if fieldnames:
            csv_writer = csv.DictWriter(
                f,
                fieldnames,
                dialect=dialect,
                delimiter=delimiter,
                quoting=quoting,
            )
            csv_writer.writeheader()
        else:
            csv_writer = csv.writer(
                f,
                dialect=dialect,
                delimiter=delimiter,
                quoting=quoting,
            )
        csv_writer.writerows(data)
