from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os
import shutil
import tarfile
import zipfile

from tqdm import tqdm

from .. import compat
from .. import data_dir as DATA_DIR
from ..io import write_http_stream


class Dataset(object):
    """
    Base class for textacy datasets.

    Args:
        name (str)
        meta (dict)

    Attributes:
        name (str)
        meta (dict)
    """

    def __init__(self, name, meta=None):
        self.name = name
        self.meta = meta or {}

    def __repr__(self):
        return 'Dataset("{}")'.format(self.name)

    @property
    def info(self):
        info = {"name": self.name}
        info.update(self.meta)
        return info

    def __iter__(self):
        raise NotImplementedError()

    def texts(self):
        raise NotImplementedError()

    def records(self):
        raise NotImplementedError()

    def download(self):
        raise NotImplementedError()


def _download(url, filename=None, dirpath=DATA_DIR, force=False):
    """
    Args:
        url (str)
        filename (str)
        dirpath (str)
        force (bool)

    Returns:
        str
    """
    if not filename:
        filename = _get_filename_from_url(url)
    filepath = os.path.join(dirpath, filename)
    if os.path.isfile(filepath) and force is False:
        logging.info(
            "file '%s' already exists and force=False; skipping download...",
            filepath,
        )
        return None
    else:
        write_http_stream(url, filepath, mode="wb", make_dirs=True)
    # TODO: check the data to make sure all is well?
    return filepath


def _get_filename_from_url(url):
    """
    Derive a filename from a URL's path.

    Args:
        url (str): URL from which to extract a filename.

    Returns:
        str: Filename in URL.
    """
    return os.path.basename(compat.urlparse(url).path)


def _unpack_archive(filepath, extract_dir=None):
    """
    Extract data from a zip or tar archive file into a directory
    (or do nothing if the file isn't an archive).

    Args:
        filepath (str): Full path to file on disk from which
            archived contents will be extracted.
        extract_dir (str): Path of the directory into which contents
            will be extracted. If not provided, the same directory
            as ``filepath`` is used.

    Returns:
        str: Path to directory of extracted contents.
    """
    # TODO: shutil.unpack_archive() when PY3-only
    if not extract_dir:
        extract_dir = os.path.dirname(filepath)
    # TODO: os.makedirs(path, exist_ok=True) when PY3-only
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
    is_zipfile = zipfile.is_zipfile(filepath)
    is_tarfile = tarfile.is_tarfile(filepath)
    if not is_zipfile and not is_tarfile:
        logging.debug("'%s' is not an archive", filepath)
        return extract_dir
    else:
        if is_zipfile:
            logging.info("extracting data from zip archive '%s'", filepath)
            with zipfile.ZipFile(filepath, mode="r") as zf:
                # zf.extractall(path=extract_dir)
                members = zf.namelist()
                with tqdm(iterable=members, total=len(members)) as pbar:
                    for member in members:
                        zf.extract(member, path=extract_dir)
                        pbar.update()
        elif is_tarfile:
            logging.info("extracting data from tar archive '%s'", filepath)
            with tarfile.open(filepath, mode="r") as tf:
                # tf.extractall(path=extract_dir)
                members = tf.getnames()
                for member in tqdm(iterable=members, total=len(members)):
                    tf.extract(member, path=extract_dir)
        src_basename = os.path.commonpath(members)
        dest_basename = os.path.basename(filepath)
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


def validate_and_clip_range(req_range, full_range, val_type=None):
    """
    Validate and clip range values, for use in filtering datasets.

    Args:
        req_range (list or tuple)
        full_range (list or tuple)
        val_type: If specified, the type or types that each value in ``req_range``
            must be instances of.

    Returns:
        tuple: Range for which null or too-small/large values have been
        clipped to the min/max valid values.

    Raises:
        TypeError
        ValueError
    """
    for range_ in (req_range, full_range):
        if not isinstance(range_, (list, tuple)):
            raise TypeError(
                "range must be a list or tuple, not {}".format(type(range_))
            )
        if len(range_) != 2:
            raise ValueError("range must have exactly two items: start and end")
    if val_type:
        for range_ in (req_range, full_range):
            for val in range_:
                if val is not None and not isinstance(val, val_type):
                    raise TypeError(
                        "range value {} must be {}, not {}".format(
                            val, val_type, type(val)
                        )
                    )
    if req_range[0] is None:
        req_range = (full_range[0], req_range[1])
    elif req_range[0] < full_range[0]:
        logging.warning(
            "start of  range %s < minimum valid value %s; clipping range ...",
            req_range[0],
            full_range[0],
        )
        req_range = (full_range[0], req_range[1])
    if req_range[1] is None:
        req_range = (req_range[0], full_range[1])
    elif req_range[1] > full_range[1]:
        logging.warning(
            "end of range %s > maximum valid value %s; clipping range ...",
            req_range[1],
            full_range[1],
        )
        req_range = (req_range[0], full_range[1])
    return tuple(req_range)
