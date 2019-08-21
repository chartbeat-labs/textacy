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

from .. import constants, utils
from ..io import utils as io_utils

LOGGER = logging.getLogger(__name__)


def download_file(url, *, filename=None, dirpath=constants.DEFAULT_DATA_DIR, force=False):
    utils.deprecated(
        "This function has been moved to `textacy.io.utils.download_file()` "
        "and is aliased here only for backwards compatibility. "
        "This alias will be removed in v0.10.0.",
        action="once",
    )
    return io_utils.download_file(url, filename=filename, dirpath=dirpath, force=force)


def get_filename_from_url(url):
    utils.deprecated(
        "This function has been moved to `textacy.io.utils.get_filename_from_url()` "
        "and is aliased here only for backwards compatibility. "
        "This alias will be removed in v0.10.0.",
        action="once",
    )
    return io_utils.get_filename_from_url(url)


def unpack_archive(filepath, *, extract_dir=None):
    utils.deprecated(
        "This function has been moved to `textacy.io.utils.unpack_archive()` "
        "and is aliased here only for backwards compatibility. "
        "This alias will be removed in v0.10.0.",
        action="once",
    )
    return io_utils.unpack_archive(filepath, extract_dir=extract_dir)


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
    filter_vals = utils.to_collection(filter_vals, vals_type, set)
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
