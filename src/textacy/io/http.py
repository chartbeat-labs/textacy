"""
:mod:`textacy.io.http`: Functions for reading data from URLs via streaming HTTP requests
and either reading it into memory or writing it directly to disk.
"""
from __future__ import annotations

import logging
from contextlib import closing
from typing import Iterable, Optional

import requests
from tqdm import tqdm

from .. import types, utils
from . import utils as io_utils


LOGGER = logging.getLogger(__name__)


def read_http_stream(
    url: str,
    *,
    lines: bool = False,
    decode_unicode: bool = False,
    chunk_size: int = 1024,
    auth: Optional[tuple[str, str]] = None,
) -> Iterable[str] | Iterable[bytes]:
    """
    Read data from ``url`` in a stream, either all at once or line-by-line.

    Args:
        url: URL to which a GET request is made for data.
        lines: If False, yield all of the data at once; otherwise, yield data line-by-line.
        decode_unicode: If True, yield data as unicode, where the encoding
            is taken from the HTTP response headers; otherwise, yield bytes.
        chunk_size: Number of bytes read into memory per chunk.
            Because decoding may occur, this is not necessarily the length of each chunk.
        auth: (username, password) pair for simple HTTP authentication required (if at all)
            to access the data at ``url``.

            .. seealso:: http://docs.python-requests.org/en/master/user/authentication/

    Yields:
        If ``lines`` is True, the next line in the response data,
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
            lines_ = r.iter_lines(chunk_size=chunk_size, decode_unicode=decode_unicode)
            for line in lines_:
                if line:
                    yield line


def write_http_stream(
    url: str,
    filepath: types.PathLike,
    *,
    mode: str = "wt",
    encoding: Optional[str] = None,
    make_dirs: bool = False,
    chunk_size: int = 1024,
    auth: Optional[tuple[str, str]] = None,
) -> None:
    """
    Download data from ``url`` in a stream, and write successive chunks
    to disk at ``filepath``.

    Args:
        url: URL to which a GET request is made for data.
        filepath: Path to file on disk to which data will be written.
        mode: Mode with which ``filepath`` is opened.
        encoding: Name of the encoding used to decode or encode the data
            in ``filepath``. Only applicable in text mode.

            .. note:: The encoding on the HTTP response is inferred from its
               headers, or set to 'utf-8' as a fall-back in the case that no
               encoding is detected. It is *not* set by ``encoding``.

        make_dirs: If True, automatically create (sub)directories if
            not already present in order to write ``filepath``.
        chunk_size: Number of bytes read into memory per chunk.
            Because decoding may occur, this is not necessarily the length of each chunk.
        auth: (username, password) pair for simple HTTP authentication required (if at all)
            to access the data at ``url``.

            .. seealso:: http://docs.python-requests.org/en/master/user/authentication/
    """
    decode_unicode = True if "t" in mode else False
    filepath = utils.to_path(filepath).resolve()
    if make_dirs is True:
        io_utils._make_dirs(filepath, mode)
    # use `closing` to ensure connection and progress bar *always* close
    with closing(requests.get(url, stream=True, auth=auth)) as r:
        LOGGER.info("downloading data from %s ...", url)
        # set fallback encoding if unable to infer from headers
        if r.encoding is None:
            r.encoding = "utf-8"
        total = int(r.headers.get("content-length", 0))
        with closing(tqdm(unit="B", unit_scale=True, total=total)) as pbar:
            with filepath.open(mode=mode, encoding=encoding) as f:
                chunks = r.iter_content(
                    chunk_size=chunk_size, decode_unicode=decode_unicode
                )
                for chunk in chunks:
                    # needed (?) to filter out "keep-alive" new chunks
                    if chunk:
                        pbar.update(len(chunk))
                        f.write(chunk)
