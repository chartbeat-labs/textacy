import collections
import csv
import io
import statistics

from spacy.parts_of_speech import ADJ, ADV, NOUN, VERB
from spacy.tokens import Doc, Span, Token

from .. import constants, utils
from ..datasets import utils as ds_utils  # TODO: move this functionality into io.utils?
from .base import Resource


NAME = "depeche_mood"
META = {
    "site_url": "http://www.depechemood.eu",
    "publication_url": "https://arxiv.org/abs/1810.03660",
    "description": "A simple tool to analyze the emotions evoked by a text.",
}
DOWNLOAD_URL = "https://github.com/marcoguerini/DepecheMood/releases/download/v2.0/DepecheMood_v2.0.zip"


class DepecheMood(Resource):
    """
    Args:
        data_dir (str or :class:`pathlib.Path`)
        lang ({"en", "it"})
        word_rep ({"token", "lemma", "lemmapos"})
        min_freq (int)
    """

    _lang_map = {"en": "english", "it": "italian"}
    _pos_map = {NOUN: "n", VERB: "v", ADJ: "a", ADV: "r"}

    def __init__(
        self,
        data_dir=constants.DEFAULT_DATA_DIR.joinpath(NAME),
        lang="en",
        word_rep="lemmapos",
        min_freq=1,
    ):
        super().__init__(NAME, meta=META)
        self.lang = lang
        self.word_rep = word_rep
        self.min_freq = min_freq
        self.data_dir = utils.to_path(data_dir).resolve()
        self._filepath = self.data_dir.joinpath(
            "DepecheMood++",
            "DepecheMood_{lang}_{word_rep}_full.tsv".format(
                lang=self._lang_map[lang], word_rep=word_rep),
        )
        self._data = None

    @property
    def filepath(self):
        """
        str: Full path on disk for the DepecheMood tsv file
        corresponding to the ``lang`` and ``word_rep``.
        """
        if self._filepath.is_file():
            return str(self._filepath)
        else:
            return None

    @property
    def data(self):
        """
        Dict[str, Dict[str, float]]: Mapping of term string (or term#POS,
        if :attr:`DepecheMood.word_rep` is "lemmapos") to the terms' normalized weights
        on a fixed set of affective dimensions (aka "emotions").
        """
        if not self._data:
            if not self.filepath:
                raise OSError(
                    "resource file {} not found;\n"
                    "has the data been downloaded yet?".format(self._filepath)
                )
            with io.open(self.filepath, mode="rt", encoding="utf-8") as csvfile:
                csv_reader = csv.reader(csvfile, delimiter="\t")
                rows = list(csv_reader)
            cols = rows[0]
            self._data = {
                row[0]: {col: float(val) for col, val in zip(cols[1:-1], row[1:-1])}
                for row in rows[1:]
                if int(row[-1]) >= self.min_freq
            }
        return self._data

    def download(self, *, force=False):
        """
        Download resource data as a zip archive file, then save it to disk
        and extract its contents under the ``data_dir`` directory.

        Args:
            force (bool): If True, download the resource, even if it already
                exists on disk under ``data_dir``.
        """
        filepath = ds_utils.download_file(
            DOWNLOAD_URL,
            filename=None,
            dirpath=self.data_dir,
            force=force,
        )
        if filepath:
            ds_utils.unpack_archive(filepath, extract_dir=None)

    def __call__(self, terms, min_weight=0.0):
        """
        Get average emotional weights of all terms in ``terms`` for which
        weights are available.

        Args:
            terms (str or Sequence[str], :class:`spacy.tokens.Token` or Sequence[Token]):
                One or more terms over which to average emotional valences.
                Note that only nouns, adjectives, adverbs, and verbs are included.
            min_weight (float): Minimum emotional valence weight for which to
                count a given term+emotion. Value must be in [0.0, 1.0)

        Returns:
            Dict[str, float]: Mapping of emotion to average weight.
        """
        if isinstance(terms, (Token, str)):
            return self._get_emotion_weights(terms)
        elif isinstance(terms, (Span, Doc, collections.abc.Sequence)):
            all_emo_weights = collections.defaultdict(list)
            for term in terms:
                emo_weights = self._get_emotion_weights(term)
                for emo, weight in emo_weights.items():
                    if weight >= min_weight:
                        all_emo_weights[emo].append(weight)
            return {
                emo: statistics.mean(weights)
                for emo, weights in all_emo_weights.items()
            }
        else:
            raise TypeError(
                "`terms` must be of type {}, not {}".format(
                    {Token, Span, Doc, str, collections.abc.Sequence}, type(terms))
            )

    def _get_emotion_weights(self, term):
        """
        Args:
            term (str or :class:`spacy.tokens.Token`)

        Returns:
            Dict[str, float]
        """
        try:
            if isinstance(term, str):
                return self.data[term]
            elif self.word_rep == "lemmapos":
                return self.data["{}#{}".format(term.lemma_, self._pos_map[term.pos])]
            elif self.word_rep == "lemma":
                return self.data[term.lemma_]
            elif self.word_rep == "token":
                return self.data[term.text]
            else:
                raise ValueError("something's wrong here...")
        except KeyError:
            return {}
