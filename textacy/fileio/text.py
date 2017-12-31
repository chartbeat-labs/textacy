from __future__ import absolute_import, print_function, unicode_literals

from .utils import open_sesame, _validate_read_mode, _validate_write_mode


def read_text(fname, mode='rt', encoding=None, lines=False):
    """
    Read the contents of a text file at ``fname``, either all at once
    or streaming line-by-line.

    Args:
        fname (str): Path to file on disk from which data will be read.
        mode (str): Mode with which ``fname`` is opened.
        encoding (str): Name of the encoding used to decode or encode the data
            in ``fname``. Only applicable in text mode.
        lines (bool): If False, all data is read in at once; otherwise, data is
            read in one line at a time.

    Yields:
        str: Next line of text to read in.

        If ``lines`` is False, wrap this output in :func:`next()` to conveniently
        access the full text.
    """
    _validate_read_mode(mode)
    with open_sesame(fname, mode=mode, encoding=encoding) as f:
        if lines is False:
            yield f.read()
        else:
            for line in f:
                yield line


def write_text(content, fname, mode='wt', encoding=None,
               make_dirs=False, lines=False):
    """
    Write text ``content`` to disk at ``fname``, either all at once
    or streaming line-by-line.

    Args:
        content (str or Iterable[str])
        fname (str): Path to file on disk to which data will be written.
        mode (str): Mode with which ``fname`` is opened.
        encoding (str): Name of the encoding used to decode or encode the data
            in ``fname``. Only applicable in text mode.
        make_dirs (bool): If True, automatically create (sub)directories if
            not already present in order to write ``fname``.
        lines (bool): If False, all data is written at once; otherwise, data is
            written to disk one line at a time.
    """
    _validate_write_mode(mode)
    with open_sesame(fname, mode=mode, encoding=encoding, make_dirs=make_dirs) as f:
        if lines is False:
            f.write(content)
        else:
            newline = '\n' if 't' in mode else b'\n'
            for line in content:
                f.write(line + newline)
