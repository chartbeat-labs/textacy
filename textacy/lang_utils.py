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

The pipeline was trained on a randomized, stratified subset of ~1.5M texts
drawn from several sources:

- **Tatoeba:** A crowd-sourced collection of ~5M sentences and their translations
  into many languages. Style is relatively informal; subject matter is a variety
  of everyday things and goings-on.
  Source: https://tatoeba.org/eng/downloads.
- **Leipzig Corpora:** A collection of corpora for many languages in the same format
  and pulling from comparable sources -- specifically, 10k Wikipedia articles from
  official database dumps and 10k news articles from either RSS feeds or web scrapes.
  Only the most recently updated version was used, when available. Style is
  relatively formal; subject matter is a variety of notable things and goings-on.
  Source: http://wortschatz.uni-leipzig.de/en/download
- **UDHR:** The UN's Universal Declaration of Human Rights document, translated
  into hundreds of languages and split into paragraphs. Style is formal;
  subject matter is fundamental human rights to be universally protected.
  Source: https://unicode.org/udhr/index.html
- **Twitter:** A collection of ~1.5k tweets in each of ~70 languages, posted in
  July 2014, with languages assigned through a combination of models and human
  annotators. Style is informal; subject matter is whatever Twitter was going on
  about back then, who could say.
  Source: https://blog.twitter.com/engineering/en_us/a/2015/evaluating-language-identification-performance.html
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import io
import logging
import operator
import os

import joblib

from . import cache
from . import compat
from . import constants
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

    def __init__(
        self,
        data_dir=os.path.join(constants.DEFAULT_DATA_DIR, "lang_identifier"),
        max_text_len=1000
    ):
        self.data_dir = data_dir
        self.filename = "textacy-lang-identifier-py{}.pkl.gz".format(2 if compat.PY2 else 3)
        self.max_text_len = max_text_len
        self._pipeline = None
        return

    @property
    def pipeline(self):
        if not self._pipeline:
            self._pipeline = self._load_pipeline()
        return self._pipeline

    def _load_pipeline(self):
        filepath = os.path.join(self.data_dir, self.filename)
        if not os.path.isfile(filepath):
            self.download()
        with io.open(filepath, mode="rb") as f:
            pipeline = joblib.load(f)
        return pipeline

    def download(self, force=False):
        """
        Download the pipeline data as a Python version-specific compressed pickle file
        and save it to disk under the :attr:`LangIdentifier.data_dir` directory.

        Args:
            force (bool): If True, download the dataset, even if it already
                exists on disk under ``data_dir``.
        """
        from .datasets.utils import download_file
        release_tag = "lang_identifier_py{py_version}_v{data_version}".format(
            py_version=2 if compat.PY2 else 3,
            data_version=1.0,
        )
        url = compat.urljoin(
            "https://github.com/bdewilde/textacy-data/releases/download/",
            release_tag + "/" + self.filename)
        filepath = download_file(
            url,
            filename=self.filename,
            dirpath=self.data_dir,
            force=force,
        )

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
                        activation="relu", solver="adam",
                        hidden_layer_sizes=(512,), alpha=0.0001, batch_size=512,
                        learning_rate_init=0.01, learning_rate="adaptive",
                        max_iter=20, early_stopping=True, tol=0.001,
                        shuffle=True, random_state=42,
                        verbose=True,
                    )
                ),
            ]
        )


lang_identifier = LangIdentifier()
identify_lang = lang_identifier.identify_lang
