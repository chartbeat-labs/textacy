"""
Dataset Utils
-------------

Shared functionality for downloading, naming, and extracting the contents
of datasets, as well as filtering for particular subsets.
"""
import logging
import os
import shutil
import tarfile
import urllib.parse
import zipfile

from tqdm import tqdm

from .. import constants
from ..utils import to_collection
from ..io import write_http_stream

LOGGER = logging.getLogger(__name__)


def download_file(url, filename=None, dirpath=constants.DEFAULT_DATA_DIR, force=False):
    """
    Download a file from ``url`` and save it to disk.

    Args:
        url (str): Web address from which to download data.
        filename (str): Name of the file to which downloaded data is saved.
            If None, a filename will be inferred from the ``url``.
        dirpath (str): Full path to the directory on disk under which
            downloaded data will be saved as ``filename``.
        force (bool): If True, download the data even if it already exists at
            ``dirpath/filename``; otherwise, only download if the data doesn't
            already exist on disk.

    Returns:
        str
    """
    if not filename:
        filename = get_filename_from_url(url)
    filepath = os.path.join(dirpath, filename)
    if os.path.isfile(filepath) and force is False:
        LOGGER.info(
            "file '%s' already exists and force=False; skipping download...",
            filepath,
        )
        return None
    else:
        write_http_stream(url, filepath, mode="wb", make_dirs=True)
    return filepath


def get_filename_from_url(url):
    """
    Derive a filename from a URL's path.

    Args:
        url (str): URL from which to extract a filename.

    Returns:
        str: Filename in URL.
    """
    return os.path.basename(urllib.parse.urlparse(urllib.parse.unquote_plus(url)).path)


def unpack_archive(filepath, extract_dir=None):
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
    os.makedirs(extract_dir, exist_ok=True)
    is_zipfile = zipfile.is_zipfile(filepath)
    is_tarfile = tarfile.is_tarfile(filepath)
    if not is_zipfile and not is_tarfile:
        LOGGER.debug("'%s' is not an archive", filepath)
        return extract_dir
    else:
        pbar_kwargs = dict(unit="files", unit_scale=True)
        if is_zipfile:
            LOGGER.info("extracting data from zip archive '%s'", filepath)
            with zipfile.ZipFile(filepath, mode="r") as zf:
                # zf.extractall(path=extract_dir)
                members = zf.namelist()
                with tqdm(iterable=members, total=len(members), **pbar_kwargs) as pbar:
                    for member in members:
                        zf.extract(member, path=extract_dir)
                        pbar.update()
        elif is_tarfile:
            LOGGER.info("extracting data from tar archive '%s'", filepath)
            with tarfile.open(filepath, mode="r") as tf:
                # tf.extractall(path=extract_dir)
                members = tf.getnames()
                for member in tqdm(iterable=members, total=len(members), **pbar_kwargs):
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


def validate_set_member_filter(filter_vals, vals_type, valid_vals=None):
    """
    Validate filter values that must be of a certain type or
    found among a set of known values.

    Args:
        filter_vals (obj or Set[obj]): Value or values to filter records by.
        vals_type (type or Tuple[type]): Type(s) of which all ``filter_vals``
            must be instances.
        valid_vals (Set[obj]): Set of valid values in which all ``filter_vals``
            must be found.

    Return:
        Set[obj]: Validated and standardized filter values.

    Raises:
        TypeError
        ValueError
    """
    filter_vals = to_collection(filter_vals, vals_type, set)
    if valid_vals is not None:
        if not all(filter_val in valid_vals for filter_val in filter_vals):
            raise ValueError(
                "not all values in filter are valid: {}".format(
                    filter_vals.difference(valid_vals)
                )
            )
    return filter_vals


def validate_and_clip_range_filter(filter_range, full_range, val_type=None):
    """
    Validate and clip range values, for use in filtering datasets.

    Args:
        filter_range (list or tuple): Range to use for filter, i.e.
            ``[start_val, end_val)`` .
        full_range (list or tuple): Full range available for filter, i.e.
            ``[min_val, max_val)`` .
        val_type: If specified, the type or types that each value in ``filter_range``
            must be instances of.

    Returns:
        tuple: Range for which null or too-small/large values have been
        clipped to the min/max valid values.

    Raises:
        TypeError
        ValueError
    """
    for range_ in (filter_range, full_range):
        if not isinstance(range_, (list, tuple)):
            raise TypeError(
                "range must be a list or tuple, not {}".format(type(range_))
            )
        if len(range_) != 2:
            raise ValueError("range must have exactly two items: start and end")
    if val_type:
        for range_ in (filter_range, full_range):
            for val in range_:
                if val is not None and not isinstance(val, val_type):
                    raise TypeError(
                        "range value {} must be {}, not {}".format(
                            val, val_type, type(val)
                        )
                    )
    if filter_range[0] is None:
        filter_range = (full_range[0], filter_range[1])
    elif filter_range[0] < full_range[0]:
        LOGGER.warning(
            "start of  range %s < minimum valid value %s; clipping range ...",
            filter_range[0],
            full_range[0],
        )
        filter_range = (full_range[0], filter_range[1])
    if filter_range[1] is None:
        filter_range = (filter_range[0], full_range[1])
    elif filter_range[1] > full_range[1]:
        LOGGER.warning(
            "end of range %s > maximum valid value %s; clipping range ...",
            filter_range[1],
            full_range[1],
        )
        filter_range = (filter_range[0], full_range[1])
    return tuple(filter_range)
