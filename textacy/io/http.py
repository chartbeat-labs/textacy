"""
HTTP
----

Functions for reading data from URLs via streaming HTTP requests and either
reading it into memory or writing it directly to disk.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import io
import logging
from contextlib import closing

import requests
from tqdm import tqdm

from .utils import _make_dirs

LOGGER = logging.getLogger(__name__)


def read_http_stream(
    url, lines=False, decode_unicode=False, chunk_size=1024, auth=None
):
    """
    Read data from ``url`` in a stream, either all at once or line-by-line.

    Args:
        url (str): URL to which a GET request is made for data.
        lines (bool): If False, yield all of the data at once; otherwise, yield
            data line-by-line.
        decode_unicode (bool): If True, yield data as unicode, where the encoding
            is taken from the HTTP response headers; otherwise, yield bytes.
        chunk_size (int): Number of bytes read into memory per chunk. Because
            decoding may occur, this is not necessarily the length of each chunk.
        auth (Tuple[str, str]): (username, password) pair for simple HTTP
            authentication required (if at all) to access the data at ``url``.

            .. seealso:: http://docs.python-requests.org/en/master/user/authentication/

    Yields:
        str or bytes: If ``lines`` is True, the next line in the response data,
        which is bytes if ``decode_unicode`` is False or unicode otherwise.
        If ``lines`` is False, yields the full response content, either as bytes
        or unicode.
    """
    # always close the connection
    with closing(requests.get(url, stream=True, auth=auth)) as r:
        # set fallback encoding if unable to infer from headers
        if r.encoding is None:
            r.encoding = "utf-8"
        if lines is False:
            if decode_unicode is True:
                yield r.text
            else:
                yield r.content
        else:
            lines = r.iter_lines(chunk_size=chunk_size, decode_unicode=decode_unicode)
            for line in lines:
                if line:
                    yield line


def write_http_stream(
    url, filepath, mode="wt", encoding=None, make_dirs=False, chunk_size=1024, auth=None
):
    """
    Download data from ``url`` in a stream, and write successive chunks
    to disk at ``filepath``.

    Args:
        url (str): URL to which a GET request is made for data.
        filepath (str): Path to file on disk to which data will be written.
        mode (str): Mode with which ``filepath`` is opened.
        encoding (str): Name of the encoding used to decode or encode the data
            in ``filepath``. Only applicable in text mode.

            .. note:: The encoding on the HTTP response is inferred from its
               headers, or set to 'utf-8' as a fall-back in the case that no
               encoding is detected. It is *not* set by ``encoding``.

        make_dirs (bool): If True, automatically create (sub)directories if
            not already present in order to write ``filepath``.
        chunk_size (int): Number of bytes read into memory per chunk. Because
            decoding may occur, this is not necessarily the length of each chunk.
        auth (Tuple[str, str]): (username, password) pair for simple HTTP
            authentication required (if at all) to access the data at ``url``.

            .. seealso:: http://docs.python-requests.org/en/master/user/authentication/
    """
    decode_unicode = True if "t" in mode else False
    if make_dirs is True:
        _make_dirs(filepath, mode)
    # use `closing` to ensure connection and progress bar *always* close
    with closing(requests.get(url, stream=True, auth=auth)) as r:
        LOGGER.info("downloading data from %s ...", url)
        # set fallback encoding if unable to infer from headers
        if r.encoding is None:
            r.encoding = "utf-8"
        total = int(r.headers.get("content-length", 0))
        with closing(tqdm(unit="B", unit_scale=True, total=total)) as pbar:
            with io.open(filepath, mode=mode, encoding=encoding) as f:
                chunks = r.iter_content(
                    chunk_size=chunk_size, decode_unicode=decode_unicode
                )
                for chunk in chunks:
                    # needed (?) to filter out "keep-alive" new chunks
                    if chunk:
                        pbar.update(len(chunk))
                        f.write(chunk)
