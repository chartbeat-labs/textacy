"""
TODO
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os
import random
try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen

import textacy
from textacy.fileio import read_json_lines, write_file


logger = logging.getLogger(__name__)

URL = 'https://s3.amazonaws.com/chartbeat-labs/bernie_and_hillary.json'
FNAME = 'bernie_and_hillary.json'
DEFAULT_DATA_DIR = os.path.join(textacy.__data_dir__, 'bnh')


def download_bernie_and_hillary(data_dir=DEFAULT_DATA_DIR):
    """
    Download the Bernie & Hillary corpus from S3, save to disk as JSON lines.

    Args:
        data_dir (str, optional): path on disk where corpus will be saved
    """
    opener = urlopen(URL)
    if opener.getcode() != 200:
        logger.error('unable to download corpus from %s', URL)
    else:
        logger.info('corpus downloaded from %s (10 MB)', URL)
    data = opener.read().decode('utf8')
    fname = os.path.join(data_dir, FNAME)
    write_file(data, fname, mode='wt', encoding=None)


def fetch_bernie_and_hillary(data_dir=DEFAULT_DATA_DIR,
                             download_if_missing=True,
                             shuffle=False):
    """
    Load the Bernie & Hillary corpus from disk (automatically downloading data
    from S3 if necessary and desired).

    Args:
        data_dir (str, optional): path on disk from which corpus will be loaded
        download_if_missing (bool, optional): if True and corpus not found on disk,
            it will be automatically downloaded from S3
        shuffle (bool, optional): if True, randomly shuffle order of documents;
            if False, documents are sorted chronologically

    Returns:
        list(dict)
    """
    fname = os.path.join(data_dir, FNAME)
    if not os.path.exists(fname):
        if download_if_missing is True:
            download_bernie_and_hillary(data_dir=data_dir)
        else:
            logger.error('unable to load corpus from %s', fname)
            return

    data = list(read_json_lines(fname, mode='rt', encoding=None))
    if shuffle is True:
        random.shuffle(data)

    return data
