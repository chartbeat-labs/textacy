"""
Language Identification
-----------------------
"""
# TODO: identify the "best" language detector available for OSS python
# in terms of prediction accuracy/precision/recall, speed, and installation
# current candidates:
# - cld2-cffi https://github.com/GregBowyer/cld2-cffi
# - pycld2 https://github.com/aboSamoor/pycld2
# - cld3 https://github.com/google/cld3 or https://github.com/Elizafox/cld3
# - langdetect https://github.com/Mimino666/langdetect
# - whatthelang https://github.com/indix/whatthelang
# - langid https://github.com/saffsd/langid.py
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import operator

try:
    from cld2 import detect as cld2_detect
except ImportError:
    pass

from . import cache
from . import compat
from . import utils

LOGGER = logging.getLogger(__name__)


def detect_lang(text):
    """
    Predict the most likely language of a text and return its 2-letter code
    (see https://cloud.google.com/translate/v2/using_rest#language-params).
    Uses the `cld2-cffi <https://pypi.python.org/pypi/cld2-cffi>`_ package;
    to take advantage of optional params, call :func:`cld2.detect()` directly.

    Args:
        text (str)

    Returns:
        str
    """
    try:
        cld2_detect
    except NameError:
        raise ImportError(
            "`cld2-cffi` must be installed to use textacy's automatic language detection; "
            "you may do so via `pip install cld2-cffi` or `pip install textacy[lang]`."
        )
    if compat.PY2:
        is_reliable, _, best_guesses = cld2_detect(
            compat.to_bytes(text), bestEffort=True
        )
    else:
        is_reliable, _, best_guesses = cld2_detect(text, bestEffort=True)
    if is_reliable is False:
        LOGGER.warning(
            "Text language detected with low confidence; best guesses: %s",
            best_guesses,
        )
    return best_guesses[0][1]


class LangIdentifier(object):
    """
    Args:
        data_dir (str)
        max_text_len (int)

    Attributes:
        pipeline (:class:`sklearn.pipeline.Pipeline`)
    """

    def __init__(self, data_dir=None, max_text_len=1000):
        self.data_dir = data_dir
        self.max_text_len = max_text_len
        self._pipeline = None
        return

    def download(self):
        raise NotImplementedError("TODO!")

    @property
    def pipeline(self):
        if not self._pipeline:
            self._pipeline = cache.load_lang_identifier(data_dir=self.data_dir)
        return self._pipeline

    def identify_lang(self, text):
        """
        Identify the most probable language identified in ``text``.

        Args:
            text (str)

        Returns:
            str: 2-letter language code of the most probable language.
        """
        text_ = utils.to_collection(text[:self.max_text_len], compat.unicode_, list)
        lang = self.pipeline.predict(text_).item()
        return lang

    def identify_topn_langs(self, text, topn=3):
        """
        Identify the ``topn`` most probable languages identified in ``text``.

        Args:
            text (str)
            topn (int)

        Returns:
            List[Tuple[str, float]]: 2-letter language code and its probability
            for the ``topn`` most probable languages.
        """
        text_ = utils.to_collection(text[:self.max_text_len], compat.unicode_, list)
        lang_probs = sorted(
            compat.zip_(self.pipeline.classes_, self.pipeline.predict_proba(text_).flat),
            key=operator.itemgetter(1),
            reverse=True,
        )[:topn]
        return [(lang.item(), prob.item()) for lang, prob in lang_probs]

    def make_pipeline(self):
        """
        Make a *new* language identification pipeline, overwriting any pre-trained
        pipeline loaded from disk under :attr:`LangIdentifier.data_dir`.
        Must be trained on (text, lang) examples before use.
        """
        import sklearn.feature_extraction
        import sklearn.neural_network
        import sklearn.pipeline

        self._pipeline = sklearn.pipeline.Pipeline(
            [
                (
                    "vectorizer",
                    sklearn.feature_extraction.text.HashingVectorizer(
                        analyzer="char_wb", ngram_range=(1, 3), lowercase=True,
                        n_features=4096, norm="l1",
                    )
                ),
                (
                    "classifier",
                    sklearn.neural_network.MLPClassifier(
                        activation="relu", solver="adam", max_iter=200,
                        shuffle=True, random_state=42,
                        learning_rate_init=0.01, learning_rate="adaptive",
                        early_stopping=True, n_iter_no_change=3, tol=0.0001,
                        hidden_layer_sizes=(512,), alpha=0.001, batch_size=512,
                        verbose=True,
                    )
                ),
            ]
        )


lang_identifier = LangIdentifier()
identify_lang = lang_identifier.identify_lang
