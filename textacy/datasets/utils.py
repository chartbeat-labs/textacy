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
    utils.deprecated(
        "This function has been moved to `textacy.utils.validate_set_members()` "
        "and is aliased here only for backwards compatibility. "
        "This alias will be removed in v0.10.0.",
        action="once",
    )
    return utils.validate_set_members(filter_vals, vals_type, valid_vals=valid_vals)


def validate_and_clip_range_filter(filter_range, full_range, val_type=None):
    utils.deprecated(
        "This function has been moved to `textacy.utils.validate_and_clip_range()` "
        "and is aliased here only for backwards compatibility. "
        "This alias will be removed in v0.10.0.",
        action="once",
    )
    return utils.validate_and_clip_range(filter_range, full_range, val_type=val_type)
