"""
Functions to load and cache language data and other NLP resources.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import csv
import io
import logging
import os
import pyphen
try:
    from urllib.request import urlopen
    from urllib.error import HTTPError
except ImportError:
    from urllib2 import urlopen
    from urllib2 import HTTPError
import warnings
import zipfile

from cachetools import cached, Cache
from cachetools.keys import hashkey
from functools import partial
import spacy

import textacy
from textacy.compat import PY2, bytes_type

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = textacy.__resources_dir__

_CACHE = {}
"""dict: key-value store used to cache datasets and such in memory"""


# TODO: maybe don't actually cache this -- it takes up a lot of RAM
# but is indeed a pain to load
@cached(Cache(1), key=partial(hashkey, 'spacy'))
def load_spacy(name, **kwargs):
    """
    Load a language-specific spaCy pipeline (collection of data, models, and
    resources) for tokenizing, tagging, parsing, etc. text; the most recent
    package loaded is cached.

    Args:
        name (str): standard 2-letter language abbreviation for a language;
            currently, spaCy supports English ('en') and German ('de')
        **kwargs: keyword arguments passed to :func:`spacy.load`; see the
            `spaCy docs <https://spacy.io/docs#english>`_ for details

            * via (str): non-default directory from which to load package data
            * vocab
            * tokenizer
            * parser
            * tagger
            * entity
            * matcher
            * serializer
            * vectors

    Returns:
        :class:`spacy.<lang>.<Language>`

    Raises:
        RuntimeError: if package can't be loaded
    """
    logger.info('Loading "%s" language spaCy pipeline', name)
    return spacy.load(name, **kwargs)


def load_spacy_pipeline(lang='en', **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn('load_spacy_pipeline() is deprecated! use load_spacy() instead.',
                      DeprecationWarning)
    return load_spacy(lang, **kwargs)


@cached(_CACHE, key=partial(hashkey, 'hyphenator'))
def load_hyphenator(lang='en'):
    """
    Load an object that hyphenates words at valid points, as used in LaTex typesetting.

    Note that while hyphenation points always fall on syllable divisions,
    not all syllable divisions are valid hyphenation points. But it's decent.

    Args:
        lang (str, optional): standard 2-letter language abbreviation;
            to get list of valid values::

                >>> import pyphen; pyphen.LANGUAGES

    Returns:
        :class:`pyphen.Pyphen()`
    """
    logger.info('Loading "%s" language hyphenator', lang)
    return pyphen.Pyphen(lang=lang)


@cached(_CACHE, key=partial(hashkey, 'depechemood'))
def load_depechemood(data_dir=None, download_if_missing=True,
                     weighting='normfreq'):
    """
    Load DepecheMood lexicon text file from disk, munge into nested dictionary
    for convenient lookup by lemma#POS. NB: English only!

    Each version of DepecheMood is built starting from word-by-document matrices
    either using raw frequencies (DepecheMood_freq.txt), normalized frequencies
    (DepecheMood_normfreq.txt) or tf-idf (DepecheMood_tfidf.txt). The files are
    tab-separated; each row contains one Lemma#PoS followed by the scores for the
    following emotions: AFRAID, AMUSED, ANGRY, ANNOYED, DONT_CARE, HAPPY, INSPIRED, SAD.

    Args:
        data_dir (str, optional): directory on disk where DepecheMood lexicon
            text files are stored, i.e. the location of the 'DepecheMood_V1.0'
            directory created when unzipping the DM dataset
        download_if_missing (bool, optional): if True and data not found on disk,
            it will be automatically downloaded and saved to disk
        weighting (str {'freq', 'normfreq', 'tfidf'}, optional): type of word
            weighting used in building DepecheMood matrix

    Returns:
        dict[dict]: top-level keys are Lemma#POS strings, values are nested dicts
            with emotion names as keys and weights as floats

    References:
        Staiano, J., & Guerini, M. (2014). "DepecheMood: a Lexicon for Emotion
        Analysis from Crowd-Annotated News". Proceedings of ACL-2014. (arXiv:1405.1605)
        Data available at https://github.com/marcoguerini/DepecheMood/releases .
    """
    # make sure data_dir is in the required format
    if data_dir is None:
        data_dir = os.path.join(DEFAULT_DATA_DIR, 'DepecheMood_V1.0')
    else:
        head, tail = os.path.split(data_dir)
        if (tail and tail != 'DepecheMood_V1.0') or head != 'DepecheMood_V1.0':
            data_dir = os.path.join(data_dir, 'DepecheMood_V1.0')
    fname = os.path.join(data_dir, 'DepecheMood_' + weighting + '.txt')
    delimiter = bytes_type('\t') if PY2 else '\t'  # HACK: Py2's csv module fail
    try:
        with io.open(fname, mode='rt') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=delimiter)
            rows = list(csvreader)
    except (OSError, IOError):
        if download_if_missing is True:
            _download_depechemood(os.path.split(data_dir)[0])
            with io.open(fname, mode='rt') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=delimiter)
                rows = list(csvreader)
        else:
            logger.exception('unable to load DepecheMood from %s', data_dir)
            raise

    logger.info('loading DepecheMood lexicon from %s', fname)
    cols = rows[0]
    return {row[0]: {cols[i]: float(row[i]) for i in range(1, 9)}
            for row in rows[1:]}


def _download_depechemood(data_dir):
    """
    Download the DepecheMood dataset from GitHub, save to disk as .txt files.

    Args:
        data_dir (str): path on disk where corpus will be saved

    Raises:
        HTTPError: if something goes wrong with the download
    """
    url = 'https://github.com/marcoguerini/DepecheMood/releases/download/v1.0/DepecheMood_V1.0.zip'
    try:
        data = urlopen(url).read()
    except HTTPError as e:
        logger.exception(
            'unable to download DepecheMood from %s; status code %s', url, e.code)
        raise
    logger.info('DepecheMood downloaded from %s (4 MB)', url)
    with zipfile.ZipFile(io.BytesIO(data)) as f:
        members = ['DepecheMood_V1.0/DepecheMood_freq.txt',
                   'DepecheMood_V1.0/DepecheMood_normfreq.txt',
                   'DepecheMood_V1.0/DepecheMood_tfidf.txt',
                   'DepecheMood_V1.0/README.txt']
        f.extractall(data_dir, members=members)
