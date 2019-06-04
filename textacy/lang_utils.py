"""
Language Identification
-----------------------

Functionality for identifying the language of a text, using a model inspired by
Google's Compact Language Detector v3 (https://github.com/google/cld3) and
implemented with ``scikit-learn``.

Model
^^^^^

Character unigrams, bigrams, and trigrams are extracted from input text, and
their frequencies of occurence within the text are counted. The full set of ngrams
are then hashed into a 4096-dimensional feature vector with values given by
the L2 norm of the counts. These features are passed into a Multi-layer Perceptron
with a single hidden layer of 512 rectified linear units and a softmax output layer
giving probabilities for ~130 different languages as ISO 639-1 language codes.

Technically, the model was implemented as a :class:`sklearn.pipeline.Pipeline`
with two steps: a :class:`sklearn.feature_extraction.text.HashingVectorizer`
for vectorizing input texts and a :class:`sklearn.neural_network.MLPClassifier`
for multi-class language classification.

Dataset
^^^^^^^

The pipeline was trained on a random subset of 2M texts drawn from two sources:

* Tatoeba: A crowd-sourced collection of ~4M sentences and their translations
  into many languages. See: https://tatoeba.org/eng/downloads.
* Wikipedia: A sample of ~200k clean, language-specific snippets collected from
  each Wikipedia's API, pulling first from articles tagged as "featured" or "good"
  then falling back on random searches. See: ``textacy/scripts/fetch_wiki_lang_snippets.py``.
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
