"""
Language identification
-----------------------

Pipeline for identifying the language of a text, using a model inspired by
Google's Compact Language Detector v3 (https://github.com/google/cld3) and
implemented with ``scikit-learn>=0.20``.

Model
^^^^^

Character unigrams, bigrams, and trigrams are extracted from input text, and
their frequencies of occurence within the text are counted. The full set of ngrams
are then hashed into a 4096-dimensional feature vector with values given by
the L2 norm of the counts. These features are passed into a Multi-layer Perceptron
with a single hidden layer of 512 rectified linear units and a softmax output layer
giving probabilities for ~140 different languages as ISO 639-1 language codes.

Technically, the model was implemented as a :class:`sklearn.pipeline.Pipeline`
with two steps: a :class:`sklearn.feature_extraction.text.HashingVectorizer`
for vectorizing input texts and a :class:`sklearn.neural_network.MLPClassifier`
for multi-class language classification.

Dataset
^^^^^^^

The pipeline was trained on a randomized, stratified subset of ~750k texts
drawn from several sources:

- **Tatoeba:** A crowd-sourced collection of sentences and their translations into
  many languages. Style is relatively informal; subject matter is a variety of
  everyday things and goings-on.
  Source: https://tatoeba.org/eng/downloads.
- **Leipzig Corpora:** A collection of corpora for many languages pulling from
  comparable sources -- specifically, 10k Wikipedia articles from official database dumps
  and 10k news articles from either RSS feeds or web scrapes, when available.
  Style is relatively formal; subject matter is a variety of notable things and goings-on.
  Source: http://wortschatz.uni-leipzig.de/en/download
- **UDHR:** The UN's Universal Declaration of Human Rights document, translated into
  hundreds of languages and split into paragraphs. Style is formal; subject matter is
  fundamental human rights to be universally protected.
  Source: https://unicode.org/udhr/index.html
- **Twitter:** A collection of tweets in each of ~70 languages, posted in July 2014,
  with languages assigned through a combination of models and human annotators.
  Style is informal; subject matter is whatever Twitter was going on about back then.
  Source: https://blog.twitter.com/engineering/en_us/a/2015/evaluating-language-identification-performance.html
- **DSLCC**: Two collections of short excerpts of journalistic texts in a handful
  of language groups that are highly similar to each other. Style is relatively formal;
  subject matter is current events.
  Source: http://ttg.uni-saarland.de/resources/DSLCC/

Performance
^^^^^^^^^^^

The trained model achieved F1 = 0.96 when (macro and micro) averaged over all languages.
A few languages have worse performance; for example, the two Norwegians ("nb" and "no"),
Bosnian ("bs") and Serbian ("sr"), and Bashkir ("ba") and Tatar ("tt") are
often confused with each other. See the textacy-data releases for more details:
https://github.com/bdewilde/textacy-data/releases/tag/lang_identifier_v1.1_sklearn_v21
"""  # noqa: E501
import logging
import operator
import urllib.parse

import joblib

from . import constants, utils
from . import io as tio

LOGGER = logging.getLogger(__name__)


class LangIdentifier:
    """
    Args:
        data_dir (str)
        max_text_len (int)

    Attributes:
        pipeline (:class:`sklearn.pipeline.Pipeline`)
    """

    def __init__(
        self,
        data_dir=constants.DEFAULT_DATA_DIR.joinpath("lang_identifier"),
        max_text_len=1000
    ):
        self._version = 1.1
        self._model_id = self._get_model_id()
        self.data_dir = utils.to_path(data_dir).resolve()
        self.filename = self._model_id + ".pkl.gz"
        self.max_text_len = max_text_len
        self._pipeline = None

    @property
    def pipeline(self):
        if not self._pipeline:
            self._pipeline = self._load_pipeline()
        return self._pipeline

    def _load_pipeline(self):
        filepath = self.data_dir.joinpath(self.filename)
        if not filepath.is_file():
            self.download()
        with filepath.open(mode="rb") as f:
            pipeline = joblib.load(f)
        return pipeline

    def _get_model_id(self):
        fstub = "lang-identifier-v{}-sklearn-v{}"
        try:
            import pkg_resources
            filename = fstub.format(
                self._version,
                pkg_resources.get_distribution("scikit-learn").version[:4]
            )
        except ImportError:
            import sklearn
            filename = fstub.format(self._version, sklearn.__version__[:4])
        return filename

    def download(self, force=False):
        """
        Download the pipeline data as a Python version-specific compressed pickle file
        and save it to disk under the :attr:`LangIdentifier.data_dir` directory.

        Args:
            force (bool): If True, download the dataset, even if it already
                exists on disk under ``data_dir``.
        """
        release_tag = self._model_id.replace("-", "_")
        url = urllib.parse.urljoin(
            "https://github.com/bdewilde/textacy-data/releases/download/",
            release_tag + "/" + self.filename)
        tio.utils.download_file(
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
        text_ = utils.to_collection(text[:self.max_text_len], str, list)
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
        text_ = utils.to_collection(text[:self.max_text_len], str, list)
        if self._is_valid(text_[0]):
            lang_probs = sorted(
                zip(self.pipeline.classes_, self.pipeline.predict_proba(text_).flat),
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
                        learning_rate_init=0.001, learning_rate="constant",
                        max_iter=15, early_stopping=True, tol=0.001,
                        shuffle=True, random_state=42,
                        verbose=True,
                    )
                ),
            ]
        )


lang_identifier = LangIdentifier()
identify_lang = lang_identifier.identify_lang
