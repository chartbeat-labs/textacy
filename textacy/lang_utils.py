"""
Language Identification
-----------------------
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import operator

from . import cache
from . import compat
from . import utils

LOGGER = logging.getLogger(__name__)


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
        if self._is_valid(text_[0]):
            lang = self.pipeline.predict(text_).item()
            return lang
        else:
            return "un"

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
        if self._is_valid(text_[0]):
            lang_probs = sorted(
                compat.zip_(self.pipeline.classes_, self.pipeline.predict_proba(text_).flat),
                key=operator.itemgetter(1),
                reverse=True,
            )[:topn]
            return [(lang.item(), prob.item()) for lang, prob in lang_probs]
        else:
            return [("un", 1.0)]

    def _is_valid(self, text):
        return any(char.isalpha() for char in text)

    def init_pipeline(self):
        """
        Initialize a *new* language identification pipeline, overwriting any
        pre-trained pipeline loaded from disk under :attr:`LangIdentifier.data_dir`.
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
                        n_features=4096, norm="l2",
                    )
                ),
                (
                    "classifier",
                    sklearn.neural_network.MLPClassifier(
                        activation="relu", solver="adam", max_iter=200,
                        hidden_layer_sizes=(512,), alpha=0.001, batch_size=512,
                        learning_rate_init=0.01, learning_rate="adaptive",
                        early_stopping=True, tol=0.0001,
                        shuffle=True, random_state=42,
                        verbose=True,
                    )
                ),
            ]
        )


lang_identifier = LangIdentifier()
identify_lang = lang_identifier.identify_lang
