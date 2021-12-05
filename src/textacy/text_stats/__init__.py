from . import basics, counts, diversity, readability, utils, _exts
from .basics import (
    n_sents,
    n_words,
    n_unique_words,
    n_chars_per_word,
    n_chars,
    n_long_words,
    n_syllables_per_word,
    n_syllables,
    n_monosyllable_words,
    n_polysyllable_words,
    entropy,
)
from .readability import (
    automated_readability_index,
    automatic_arabic_readability_index,
    coleman_liau_index,
    flesch_kincaid_grade_level,
    flesch_reading_ease,
    gulpease_index,
    gunning_fog_index,
    lix,
    mu_legibility_index,
    perspicuity_index,
    smog_index,
    wiener_sachtextformel,
)
from .api import TextStats
from .utils import load_hyphenator
