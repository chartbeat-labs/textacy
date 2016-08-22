"""
Collection of lexicon-based methods for characterizing texts by sentiment,
emotional valence, etc.
"""
from collections import defaultdict
from spacy.parts_of_speech import ADJ, ADV, NOUN, VERB

from textacy import data

# TODO: Do something smarter for averaging emotional valences.


def emotional_valence(words, threshold=0.0, dm_data_dir=None, dm_weighting='normfreq'):
    """
    Get average emotional valence over all words for the following emotions:
    AFRAID, AMUSED, ANGRY, ANNOYED, DONT_CARE, HAPPY, INSPIRED, SAD.

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
    """
    dm = data.load_depechemood(data_dir=dm_data_dir, weighting=dm_weighting)
    pos_to_letter = {NOUN: 'n', ADJ: 'a', ADV: 'r', VERB: 'v'}
    emo_matches = defaultdict(int)
    emo_scores = defaultdict(float)
    for word in words:
        if word.pos in pos_to_letter:
            lemma_pos = word.lemma_ + '#' + pos_to_letter[word.pos]
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
