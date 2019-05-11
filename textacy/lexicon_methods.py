"""
Lexicon Methods
---------------

Collection of lexicon-based methods for characterizing texts by sentiment,
emotional valence, etc.

This module is a bit of an orphan right now...
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import io
import logging
import os
import zipfile

import requests
from spacy.parts_of_speech import ADJ, ADV, NOUN, VERB

from . import cache
from . import constants

LOGGER = logging.getLogger(__name__)

# TODO: Do something smarter for averaging emotional valences?


def emotional_valence(words, threshold=0.0, dm_data_dir=None, dm_weighting="normfreq"):
    """
    Get average emotional valence over all words for the following emotions --
    AFRAID, AMUSED, ANGRY, ANNOYED, DONT_CARE, HAPPY, INSPIRED, SAD -- using
    the [DepecheMood]_ dataset.

    Args:
        words (List[``spacy.Token``]): list of words for which to get
            average emotional valence; note that only nouns, adjectives, adverbs,
            and verbs will be counted
        threshold (float): minimum emotional valence score for which to
            count a given word for a given emotion; value must be in [0.0, 1.0)
        dm_data_dir (str): full path to directory where DepecheMood data
            is saved on disk
        dm_weighting ({'freq', 'normfreq', 'tfidf'}): type of word
            weighting used in building DepecheMood matrix

    Returns:
        dict: mapping of emotion (str) to average valence score (float)

    References:
        .. [DepecheMood] Staiano and Guerini. DepecheMood: a Lexicon for Emotion
           Analysis from Crowd-Annotated News. 2014.
           Data available at https://github.com/marcoguerini/DepecheMood/releases

    See Also:
        :func:`cache.load_depechemood() <textacy.cache.load_depechemood>`
    """
    dm = cache.load_depechemood(data_dir=dm_data_dir, weighting=dm_weighting)
    pos_to_letter = {NOUN: "n", ADJ: "a", ADV: "r", VERB: "v"}
    emo_matches = collections.defaultdict(int)
    emo_scores = collections.defaultdict(float)
    for word in words:
        if word.pos in pos_to_letter:
            lemma_pos = word.lemma_ + "#" + pos_to_letter[word.pos]
            try:
                for emo, score in dm[lemma_pos].items():
                    if score > threshold:
                        emo_matches[emo] += 1
                        emo_scores[emo] += score
            except KeyError:
                continue

    for emo in emo_scores:
        emo_scores[emo] /= emo_matches[emo]

    return emo_scores


def download_depechemood(data_dir=None, force=False):
    """
    Download the DepecheMood dataset from GitHub, save to disk as .txt files.

    Args:
        data_dir (str): Path to directory on disk where data files are to be saved.
        force (bool): If True, download the data, even if it already exists
            on disk under ``data_dir``.
    """
    if data_dir is None:
        data_dir = os.path.join(constants.DEFAULT_DATA_DIR, "depechemood")
    if os.path.exists(os.path.join(data_dir, "DepecheMood_V1.0")) and force is False:
        LOGGER.warning(
            "DepecheMood data already exists in %s; skipping download...", data_dir
        )
        return
    url = "https://github.com/marcoguerini/DepecheMood/releases/download/v1.0/DepecheMood_V1.0.zip"
    try:
        response = requests.get(url)
    except requests.exceptions.RequestException:
        LOGGER.exception(
            "Unable to download DepecheMood from %s; URL status code = %s",
            url,
            response.status_code,
        )
        raise
    with zipfile.ZipFile(io.BytesIO(response.content)) as f:
        members = [
            "DepecheMood_V1.0/DepecheMood_freq.txt",
            "DepecheMood_V1.0/DepecheMood_normfreq.txt",
            "DepecheMood_V1.0/DepecheMood_tfidf.txt",
            "DepecheMood_V1.0/README.txt",
        ]
        f.extractall(data_dir, members=members)
    LOGGER.info(
        "Downloaded DepecheMood (4MB) from %s and wrote it to %s", url, data_dir
    )
