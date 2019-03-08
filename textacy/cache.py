"""
Cache
-----

Functions to load and cache language data and other NLP resources. Loading data
from disk can be slow; let's just do it once and forget about it. :)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import csv
import functools
import io
import inspect
import logging
import os
import sys

import spacy
from cachetools import cached, LRUCache
from cachetools.keys import hashkey

from . import compat
from . import data_dir as DEFAULT_DATA_DIR

LOGGER = logging.getLogger(__name__)


def _get_size(obj, seen=None):
    """
    Recursively find the actual size of an object, in bytes.

    Taken as-is (with a tweak in function name) from https://github.com/bosswissam/pysize.
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if hasattr(obj, "__dict__"):
        for cls in obj.__class__.__mro__:
            if "__dict__" in cls.__dict__:
                d = cls.__dict__["__dict__"]
                if inspect.isgetsetdescriptor(d) or inspect.ismemberdescriptor(d):
                    size += _get_size(obj.__dict__, seen)
                break
    if isinstance(obj, dict):
        size += sum((_get_size(v, seen) for v in obj.values()))
        size += sum((_get_size(k, seen) for k in obj.keys()))
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        size += sum((_get_size(i, seen) for i in obj))
    return size


LRU_CACHE = LRUCache(
    os.environ.get("TEXTACY_MAX_CACHE_SIZE", 2147483648), getsizeof=_get_size
)
""":class:`cachetools.LRUCache`: Least Recently Used (LRU) cache for loaded data.

The max cache size may be set by the `TEXTACY_MAX_CACHE_SIZE` environment variable,
where the value must be an integer (in bytes). Otherwise, the max size is 2GB.
"""


def clear():
    """Clear textacy's cache of loaded data."""
    global LRU_CACHE
    LRU_CACHE.clear()


@cached(LRU_CACHE, key=functools.partial(hashkey, "spacy"))
def load_spacy(name, disable=None):
    """
    Load a spaCy pipeline (model weights as binary data, ordered sequence of
    component functions, and language-specific data) for tokenizing and annotating
    text. An LRU cache saves pipelines in memory, up to 2GB.

    Args:
        name (str or :class:`pathlib.Path`): spaCy model to load, i.e. a shortcut
            link, full package name, or path to model directory.
        disable (Tuple[str]): Names of pipeline components to disable, if any.

            .. note:: Although spaCy's API specifies this argument as a list,
               here we require a tuple. Pipelines are stored in the LRU cache
               with unique identifiers generated from the hash of the function
               name and args, and lists aren't hashable.

    Returns:
        ``spacy.<lang>.<Language>``: A Language object with the loaded model.

    .. seealso:: https://spacy.io/api/top-level#spacy.load
    """
    if disable is None:
        disable = []
    LOGGER.debug('Loading "%s" spaCy pipeline', name)
    return spacy.load(name, disable=disable)


@cached(LRU_CACHE, key=functools.partial(hashkey, "hyphenator"))
def load_hyphenator(lang):
    """
    Load an object that hyphenates words at valid points, as used in LaTex typesetting.

    Args:
        lang (str): Standard 2-letter language abbreviation. To get a list of
            valid values::

                >>> import pyphen; pyphen.LANGUAGES

    Returns:
        :class:`pyphen.Pyphen()`

    Note:
        While hyphenation points always fall on syllable divisions,
        not all syllable divisions are valid hyphenation points. But it's decent.
    """
    import pyphen

    LOGGER.debug('Loading "%s" language hyphenator', lang)
    return pyphen.Pyphen(lang=lang)


@cached(LRU_CACHE, key=functools.partial(hashkey, "depechemood"))
def load_depechemood(data_dir=None, weighting="normfreq"):
    """
    Load DepecheMood lexicon text file from disk, munge into nested dictionary
    for convenient lookup by lemma#POS. NB: English only!

    Each version of DepecheMood is built starting from word-by-document matrices
    either using raw frequencies (DepecheMood_freq.txt), normalized frequencies
    (DepecheMood_normfreq.txt) or tf-idf (DepecheMood_tfidf.txt). The files are
    tab-separated; each row contains one Lemma#PoS followed by the scores for the
    following emotions: AFRAID, AMUSED, ANGRY, ANNOYED, DONT_CARE, HAPPY, INSPIRED, SAD.

    Args:
        data_dir (str): Directory on disk where DepecheMood lexicon text fields,
            i.e. the location of the 'DepecheMood_V1.0' directory created when
            unzipping the DM dataset.
        weighting ({'freq', 'normfreq', 'tfidf'}): Type of word
            weighting used in building DepecheMood matrix.

    Returns:
        Dict[dict]: Top-level keys are Lemma#POS strings, values are nested dicts
        with emotion names as keys and weights as floats.

    References:
        Staiano, J., & Guerini, M. (2014). "DepecheMood: a Lexicon for Emotion
        Analysis from Crowd-Annotated News". Proceedings of ACL-2014. (arXiv:1405.1605)
        Data available at https://github.com/marcoguerini/DepecheMood/releases .

    See Also:
        :func:`download_depechemood <textacy.lexicon_methods.download_depechemood>`
    """
    if data_dir is None:
        data_dir = os.path.join(DEFAULT_DATA_DIR, "depechemood", "DepecheMood_V1.0")
    fname = os.path.join(
        data_dir, "DepecheMood_{weighting}.txt".format(weighting=weighting)
    )
    delimiter = (
        compat.bytes_("\t") if compat.is_python2 else "\t"
    )  # HACK: Py2's csv module fail
    try:
        with io.open(fname, mode="rt") as csvfile:
            csvreader = csv.reader(csvfile, delimiter=delimiter)
            rows = list(csvreader)
    except (OSError, IOError):
        LOGGER.exception(
            "Unable to load DepecheMood from %s."
            "\n\nHave you downloaded the data? If not, you can use the "
            "`textacy.lexicon_methods.download_depechemood()` function."
            "\n\nIf so, have you given the correct `data_dir`? The directory "
            "should have a `DepecheMood_V1.0` subdirectory, within which are "
            "three text files and a README.",
            data_dir,
        )
        raise
    LOGGER.debug("Loading DepecheMood lexicon from %s", fname)
    cols = rows[0]
    return {
        row[0]: {cols[i]: float(row[i]) for i in compat.range_(1, 9)}
        for row in rows[1:]
    }
