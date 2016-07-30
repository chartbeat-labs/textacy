"""
Bernie & Hillary Corpus
-----------------------

Download to and load from disk a corpus of (all?) speeches given by Bernie Sanders
and Hillary Clinton on the floor of Congress between January 1996 and February 2016.

The corpus contains just over 3000 documents: 2200 by Bernie, 800 by Hillary
(Bernie has been a member of Congress significantly longer than Hillary was).
It is comprised of about 1.9 million tokens. Each document contains 6 fields:

    * text: full(?) text of the speech
    * title: title of the speech, in all caps
    * date: date on which the speech was given, as an ISO-standard string
    * speaker: name of the speaker; either 'Bernard Sanders' or 'Hillary Clinton'
    * congress: number of the Congress in which the speech was given; ranges
      continuously between 104 and 114
    * chamber: chamber of Congress in which the speech was given; almost all are
      either 'House' or 'Senate', with a small number of 'Extensions'

The source for this corpus is the Sunlight Foundation's
`Capitol Words API <http://sunlightlabs.github.io/Capitol-Words/>`_.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os
import random
try:
    from urllib.request import urlopen
    from urllib.error import HTTPError
except ImportError:
    from urllib2 import urlopen
    from urllib2 import HTTPError
import warnings

import textacy
from textacy.fileio import read_json_lines, write_file


logger = logging.getLogger(__name__)

URL = 'https://s3.amazonaws.com/chartbeat-labs/bernie_and_hillary.json'
FNAME = 'bernie_and_hillary.json'
DEFAULT_DATA_DIR = os.path.join(textacy.__resources_dir__, 'bernie_and_hillary')


def _download_bernie_and_hillary(data_dir):
    """
    Download the Bernie & Hillary corpus from S3, save to disk as JSON lines.

    Args:
        data_dir (str): path on disk where corpus will be saved

    Raises:
        HTTPError: if something goes wrong with the download
    """
    try:
        data = urlopen(URL).read()
    except HTTPError as e:
        logger.exception(
            'unable to download corpus from %s; status code %s', URL, e.code)
        raise
    logger.info('corpus downloaded from %s (10 MB)', URL)
    data = data.decode('utf8')
    fname = os.path.join(data_dir, FNAME)
    write_file(data, fname, mode='wt', encoding=None)


def fetch_bernie_and_hillary(data_dir=None,
                             download_if_missing=True,
                             shuffle=False):
    """
    Load the Bernie & Hillary corpus from disk (automatically downloading data
    from S3 if necessary and desired).

    Args:
        data_dir (str): path on disk from which corpus will be loaded;
            if None, textacy's default data_dir is used (optional)
        download_if_missing (bool): if True and corpus not found on disk, it will
            be automatically downloaded from S3 and saved to disk (optional)
        shuffle (bool): if True, randomly shuffle order of documents;
            if False, documents are sorted chronologically (optional)

    Returns:
        list[dict]: each item in list corresponds to a speech document

    Raises:
        IOError: if file is not found on disk and `download_if_missing` is False
        HTTPError: if file is not found on disk, `download_if_missing` is True,
            and something goes wrong with the download

    .. warn:: The Bernie & Hillary corpus has been deprecated! Use the newer and
        more comprehensive CapitolWords corpus instead. To recreate B&H, filter
        CapitolWords speeches by `speaker_name={'Bernie Sanders', 'Hillary Clinton'}`.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('always', DeprecationWarning)
        msg = """
            The Bernie & Hillary corpus has been deprecated! Use the newer and
            more comprehensive CapitolWords corpus instead. To recreate B&H,
            filter CapitolWords speeches by `speaker_name={'Bernie Sanders', 'Hillary Clinton'}`.
            """
        warnings.warn(msg.strip(),
                      DeprecationWarning)
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    fname = os.path.join(data_dir, FNAME)
    try:
        data = list(read_json_lines(fname, mode='rt', encoding=None))
    except (OSError, IOError):
        if download_if_missing is True:
            _download_bernie_and_hillary(data_dir=data_dir)
            data = list(read_json_lines(fname, mode='rt', encoding=None))
        else:
            logger.exception('unable to load corpus from %s', fname)
            raise
    logger.info('loading corpus from %s', fname)

    if shuffle is True:
        random.shuffle(data)

    return data
