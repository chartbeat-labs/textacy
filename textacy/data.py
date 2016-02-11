"""
Functions to load and cache language data and other NLP resources.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import csv
import logging
import json
import os
import pyphen

from cachetools import cached, Cache, hashkey
from functools import partial


logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources')

_CACHE = {}
"""dict: key-value store used to cache datasets and such in memory"""


# TODO: maybe don't actually cache this -- it takes up a lot of RAM
# but is indeed a pain to load
@cached(Cache(1), key=partial(hashkey, 'spacy_pipeline'))
def load_spacy_pipeline(lang='en', **kwargs):
    """
    Load a language-specific pipeline (collection of data, models, and resources)
    via Spacy for tokenizing, tagging, parsing, etc. raw text.

    Args:
        lang (str {'en'}, optional): standard 2-letter language abbreviation
        **kwargs: keyword arguments to pass to Spacy pipeline instantiation;
            see `Spacy's documentation <https://spacy.io/docs#api>`_

    Returns:
        :class:`spacy.<lang>.<Language>`

    Raises:
        ValueError: if `lang` not equal to 'en' (more languages coming?!?)
    """
    logger.info('Loading "%s" language Spacy pipeline', lang)
    if lang == 'en':
        from spacy.en import English
        return English(**kwargs)
    # TODO: uncomment these whenever spacy makes them available...
    # elif lang == 'de':
    #     from spacy.de import German
    #     return German(**kwargs)
    # elif lang == 'it':
    #     from spacy.it import Italian
    #     return Italian(**kwargs)
    # elif lang == 'fi':
    #     from spacy.fi import Finnish
    #     return Finnish(**kwargs)
    else:
        msg = 'spacy does not currently support lang "{}"'.format(lang)
        raise ValueError(msg)


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
def load_depechemood(data_dir=None, weighting='normfreq'):
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
            text files are stored
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
    if data_dir is None:
        data_dir = os.path.join(DEFAULT_DATA_DIR, 'DepecheMood_V1.0')
    fname = os.path.join(data_dir, 'DepecheMood_' + weighting + '.txt')
    # let's make sure this file exists...
    _ = os.path.isfile(fname)
    logger.info('Loading DepecheMood lexicon from %s', fname)
    with open(fname, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t')
        rows = list(csvreader)
    cols = rows[0]
    return {row[0]: {cols[i]: float(row[i]) for i in range(1, 9)}
            for row in rows[1:]}
