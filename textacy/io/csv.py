"""
CSV
---

Functions for reading from and writing to disk records in CSV format, where
CSVs may be delimited not only by commas (the default) but tabs, pipes, and
other valid one-char delimiters.
"""
from __future__ import absolute_import, print_function, unicode_literals

from .. import compat
from .utils import open_sesame


def read_csv(fname, encoding=None, fieldnames=None, dialect="excel", delimiter=","):
    """
    Read the contents of a CSV file at ``fname``, streaming line-by-line,
    where each line is a list of strings and/or floats whose values
    are separated by ``delimiter``.

    Args:
        fname (str): Path to file on disk from which data will be read.
        encoding (str): Name of the encoding used to decode or encode the data
            in ``fname``.
        fieldnames (List[str] or 'infer'): If specified, gives names for columns
            of values, which are used as keys in an ordered dictionary representation
            of each line's data. If 'infer', the first kB of data is analyzed
            to make a guess about whether the first row is a header of column
            names, and if so, those names are used as keys. If None, no column
            names are used, and each line is returned as a list of strings/floats.
        dialect (str): Grouping of formatting parameters that determine how
            the data is parsed when reading/writing. If 'infer', the first kB
            of data is analyzed to get a best guess for the correct dialect.
        delimiter (str): 1-character string used to separate fields in a row.

    Yields:
        List[obj]: Next row, whose elements are strings and/or floats.
        If ``fieldnames`` is None or 'infer' doesn't detect a header row.

        *or*

        Dict[str, obj]: Next row, as an ordered dictionary of (key, value) pairs,
        where keys are column names and values are the corresponding strings
        and/or floats. If ``fieldnames`` is a list of column names or 'infer'
        detects a header row.

    See Also:
        https://docs.python.org/3/library/csv.html#csv.reader
    """
    has_header = False
    with open_sesame(fname, mode="rt", encoding=encoding, newline="") as f:
        if dialect == "infer" or fieldnames == "infer":
            sniffer = compat.csv.Sniffer()
            # add pipes to the list of preferred delimiters, and put spaces last
            sniffer.preferred = [",", "\t", "|", ";", ":", " "]
            sample = "".join(f.readline() for _ in range(5))  # f.read(1024)
            if dialect == "infer":
                dialect = sniffer.sniff(sample)
            if fieldnames == "infer":
                has_header = sniffer.has_header(sample)
            f.seek(0)
        if has_header is True:
            csv_reader = compat.csv.DictReader(
                f,
                fieldnames=None,
                dialect=dialect,
                delimiter=delimiter,
                quoting=compat.csv.QUOTE_NONNUMERIC,
            )
        elif fieldnames:
            csv_reader = compat.csv.DictReader(
                f,
                fieldnames=fieldnames,
                dialect=dialect,
                delimiter=delimiter,
                quoting=compat.csv.QUOTE_NONNUMERIC,
            )
            first_row = next(csv_reader)
            # is the first row a header with same values as fieldnames?
            # if not, we should yield the row as usual
            if not all(key == value for key, value in first_row.items()):
                yield first_row
        else:
            csv_reader = compat.csv.reader(
                f,
                dialect=dialect,
                delimiter=delimiter,
                quoting=compat.csv.QUOTE_NONNUMERIC,
            )
        for row in csv_reader:
            yield row


def write_csv(
    data,
    fname,
    encoding=None,
    make_dirs=False,
    fieldnames=None,
    dialect="excel",
    delimiter=",",
):
    """
    Write rows of ``data`` to disk at ``fname``, where each row is an iterable
    or a dictionary of strings and/or numbers, written to one line with values
    separated by ``delimiter``.

    Args:
        data (Iterable[Iterable] or Iterable[dict]): If ``fieldnames`` is None,
            an iterable of iterables of strings and/or numbers to write to disk;
            for example::

                [['That was a great movie!', 0.9],
                 ['The movie was okay, I guess.', 0.2],
                 ['Worst. Movie. Ever.', -1.0]]

            If ``fieldnames`` is specified, an iterable of dictionaries with
            string and/or number values to write to disk; for example::

                [{'text': 'That was a great movie!', 'score': 0.9},
                 {'text': 'The movie was okay, I guess.', 'score': 0.2},
                 {'text': 'Worst. Movie. Ever.', 'score': -1.0}]

        fname (str): Path to file on disk to which data will be written.
        encoding (str): Name of the encoding used to decode or encode the data
            in ``fname``.
        make_dirs (bool): If True, automatically create (sub)directories if
            not already present in order to write ``fname``.
        fieldnames (List[str]): Sequence of keys that identify the order in which
            values in each rows' dictionary is written to ``fname``. These are
            included in ``fname`` as a header row of column names.

            .. note:: Only specify this if ``data`` is an iterable of dictionaries.

        dialect (str): Grouping of formatting parameters that determine how
            the data is parsed when reading/writing.
        delimiter (str): 1-character string used to separate fields in a row.

    See Also:
        https://docs.python.org/3/library/csv.html#csv.writer
    """
    with open_sesame(
        fname, mode="wt", newline="", encoding=encoding, make_dirs=make_dirs
    ) as f:
        if fieldnames:
            csv_writer = compat.csv.DictWriter(
                f,
                fieldnames,
                dialect=dialect,
                delimiter=delimiter,
                quoting=compat.csv.QUOTE_NONNUMERIC,
            )
            csv_writer.writeheader()
        else:
            csv_writer = compat.csv.writer(
                f,
                dialect=dialect,
                delimiter=delimiter,
                quoting=compat.csv.QUOTE_NONNUMERIC,
            )
        csv_writer.writerows(data)
