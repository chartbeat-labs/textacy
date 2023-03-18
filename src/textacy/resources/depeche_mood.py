"""
DepecheMood
-----------

DepecheMood is a high-quality and high-coverage emotion lexicon for English and Italian
text, mapping individual terms to their emotional valences. These word-emotion weights
are inferred from crowd-sourced datasets of emotionally tagged news articles
(rappler.com for English, corriere.it for Italian).

English terms are assigned weights to eight emotions:

    - AFRAID
    - AMUSED
    - ANGRY
    - ANNOYED
    - DONT_CARE
    - HAPPY
    - INSPIRED
    - SAD

Italian terms are assigned weights to five emotions:

    - DIVERTITO (~amused)
    - INDIGNATO (~annoyed)
    - PREOCCUPATO (~afraid)
    - SODDISFATTO (~happy)
    - TRISTE (~sad)
"""
from __future__ import annotations

import collections
import csv
import io
import statistics
from typing import Any, ClassVar, Literal, Optional, Sequence

from spacy.parts_of_speech import ADJ, ADV, NOUN, VERB
from spacy.tokens import Doc, Span, Token

from .. import constants
from .. import io as tio
from .. import types, utils
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
    Interface to DepecheMood, an emotion lexicon for English and Italian text.

    Download the data (one time only!), and save its contents to disk::

        >>> import textacy.resources
        >>> rs = textacy.resources.DepecheMood(lang="en", word_rep="lemmapos")
        >>> rs.download()
        >>> rs.info
        {'name': 'depeche_mood',
         'site_url': 'http://www.depechemood.eu',
         'publication_url': 'https://arxiv.org/abs/1810.03660',
         'description': 'A simple tool to analyze the emotions evoked by a text.'}

    Access emotional valences for individual terms::

        >>> rs.get_emotional_valence("disease#n")
        {'AFRAID': 0.37093526222120465,
         'AMUSED': 0.06953745082761113,
         'ANGRY': 0.06979683067736414,
         'ANNOYED': 0.06465401081252636,
         'DONT_CARE': 0.07080580707440012,
         'HAPPY': 0.07537324330608403,
         'INSPIRED': 0.13394731320662606,
         'SAD': 0.14495008187418348}
        >>> rs.get_emotional_valence("heal#v")
        {'AFRAID': 0.060450319886187334,
         'AMUSED': 0.09284046387491741,
         'ANGRY': 0.06207816933776029,
         'ANNOYED': 0.10027622719958346,
         'DONT_CARE': 0.11259594401785,
         'HAPPY': 0.09946106491457314,
         'INSPIRED': 0.37794768332634626,
         'SAD': 0.09435012744278205}

    When passing multiple terms in the form of a list[str] or ``Span`` or ``Doc``,
    emotion weights are averaged over all terms for which weights are available::

        >>> rs.get_emotional_valence(["disease#n", "heal#v"])
        {'AFRAID': 0.215692791053696,
         'AMUSED': 0.08118895735126427,
         'ANGRY': 0.06593750000756221,
         'ANNOYED': 0.08246511900605491,
         'DONT_CARE': 0.09170087554612506,
         'HAPPY': 0.08741715411032858,
         'INSPIRED': 0.25594749826648616,
         'SAD': 0.11965010465848278}
        >>> text = "The acting was sweet and amazing, but the plot was dumb and terrible."
        >>> doc = textacy.make_spacy_doc(text, lang="en")
        >>> rs.get_emotional_valence(doc)
        {'AFRAID': 0.05272350876803627,
         'AMUSED': 0.13725054992595098,
         'ANGRY': 0.15787016147081184,
         'ANNOYED': 0.1398733360688608,
         'DONT_CARE': 0.14356943460620503,
         'HAPPY': 0.11923217912716871,
         'INSPIRED': 0.17880214720077342,
         'SAD': 0.07067868283219296}
        >>> rs.get_emotional_valence(doc[0:6])  # the acting was sweet and amazing
        {'AFRAID': 0.039790959333750785,
         'AMUSED': 0.1346884072825313,
         'ANGRY': 0.1373596223131593,
         'ANNOYED': 0.11391999698695347,
         'DONT_CARE': 0.1574819173485831,
         'HAPPY': 0.1552521762333925,
         'INSPIRED': 0.21232264216449326,
         'SAD': 0.049184278337136296}

    For good measure, here's how Italian w/o POS-tagged words looks::

        >>> rs = textacy.resources.DepecheMood(lang="it", word_rep="lemma")
        >>> rs.get_emotional_valence("amore")
        {'INDIGNATO': 0.11451408951814121,
         'PREOCCUPATO': 0.1323655108545536,
         'TRISTE': 0.18249663560400609,
         'DIVERTITO': 0.33558928569110086,
         'SODDISFATTO': 0.23503447833219815}

    Args:
        data_dir: Path to directory on disk under which resource data is stored,
            i.e. ``/path/to/data_dir/depeche_mood``.
        lang: Standard two-letter code for the language of terms
            for which emotional valences are to be retrieved.
        word_rep: Level of text processing used in computing terms' emotion weights.
            "token" => tokenization only;
            "lemma" => tokenization and lemmatization;
            "lemmapos" => tokenization, lemmatization, and part-of-speech tagging.
        min_freq: Minimum number of times that a given term must have appeared
            in the source dataset for it to be included in the emotion weights dict.
            This can be used to remove noisy terms at the expense of reducing coverage.
            Researchers observed peak performance at 10, but anywhere between
            1 and 20 is reasonable.
    """

    _lang_map: ClassVar[dict[str, str]] = {"en": "english", "it": "italian"}
    _pos_map: ClassVar[dict[Any, str]] = {NOUN: "n", VERB: "v", ADJ: "a", ADV: "r"}
    _word_reps: ClassVar[tuple[str, str, str]] = ("token", "lemma", "lemmapos")

    def __init__(
        self,
        data_dir: types.PathLike = constants.DEFAULT_DATA_DIR.joinpath(NAME),
        lang: Literal["en", "it"] = "en",
        word_rep: Literal["token", "lemma", "lemmapos"] = "lemmapos",
        min_freq: int = 3,
    ):
        super().__init__(NAME, meta=META)
        if lang not in self._lang_map:
            raise ValueError(
                "lang='{}' is invalid; valid options are {}".format(
                    lang, sorted(self._lang_map.keys())
                )
            )
        if word_rep not in self._word_reps:
            raise ValueError(
                "word_rep='{}' is invalid; valid options are {}".format(
                    word_rep, self._word_reps
                )
            )
        self.lang = lang
        self.word_rep = word_rep
        self.min_freq = min_freq
        self.data_dir = utils.to_path(data_dir).resolve()
        self._filepath = self.data_dir.joinpath(
            "DepecheMood++",
            "DepecheMood_{lang}_{word_rep}_full.tsv".format(
                lang=self._lang_map[lang], word_rep=word_rep
            ),
        )
        self._weights: Optional[dict[str, dict[str, float]]] = None

    @property
    def filepath(self) -> Optional[str]:
        """
        Full path on disk for the DepecheMood tsv file
        corresponding to the ``lang`` and ``word_rep``.
        """
        if self._filepath.is_file():
            return str(self._filepath)
        else:
            return None

    @property
    def weights(self) -> dict[str, dict[str, float]]:
        """
        Mapping of term string (or term#POS, if :attr:`DepecheMood.word_rep` is "lemmapos")
        to the terms' normalized weights on a fixed set of affective dimensions
        (aka "emotions").
        """
        if not self._weights:
            if not self.filepath:
                raise OSError(
                    "resource file {} not found;\n"
                    "has the data been downloaded yet?".format(self._filepath)
                )
            with io.open(self.filepath, mode="rt", encoding="utf-8") as csvfile:
                csv_reader = csv.reader(csvfile, delimiter="\t")
                rows = list(csv_reader)
            cols = rows[0]
            self._weights = {
                row[0]: {col: float(val) for col, val in zip(cols[1:-1], row[1:-1])}
                for row in rows[1:]
                if int(row[-1]) >= self.min_freq
            }
        return self._weights

    def download(self, *, force: bool = False):
        """
        Download resource data as a zip archive file, then save it to disk
        and extract its contents under the ``data_dir`` directory.

        Args:
            force (bool): If True, download the resource, even if it already
                exists on disk under ``data_dir``.
        """
        filepath = tio.download_file(
            DOWNLOAD_URL,
            filename=None,
            dirpath=self.data_dir,
            force=force,
        )
        if filepath:
            tio.unpack_archive(filepath, extract_dir=None)

    def get_emotional_valence(
        self, terms: str | Token | Sequence[str] | Sequence[Token]
    ) -> dict[str, float]:
        """
        Get average emotional valence over all terms in ``terms`` for which
        emotion weights are available.

        Args:
            terms: One or more terms over which to average emotional valences.
                Note that only nouns, adjectives, adverbs, and verbs are included.

                .. note:: If the resource was initialized with ``word_rep="lemmapos"``,
                   then string terms must have matching parts-of-speech appended to them
                   like TERM#POS. Only "n" => noun, "v" => verb, "a" => adjective, and
                   "r" => adverb are included in the data.

        Returns:
            Mapping of emotion to average weight.
        """
        if isinstance(terms, (Token, str)):
            return self._get_term_emotional_valence(terms)
        elif isinstance(terms, (Span, Doc, collections.abc.Sequence)):
            return self._get_terms_emotional_valence(terms)
        else:
            raise TypeError(
                "`terms` must be of type {}, not {}".format(
                    {Token, Span, Doc, str, collections.abc.Sequence}, type(terms)
                )
            )

    def _get_term_emotional_valence(self, term: str | Token) -> dict[str, float]:
        try:
            if isinstance(term, str):
                return self.weights[term]
            elif isinstance(term, Token):
                if self.word_rep == "lemmapos":
                    return self.weights[
                        "{}#{}".format(term.lemma_, self._pos_map[term.pos])
                    ]
                elif self.word_rep == "lemma":
                    return self.weights[term.lemma_]
                else:  # word_rep == "token"
                    return self.weights[term.text]
            else:
                raise TypeError(
                    "`term` must be of type {}, not {}".format({str, Token}, type(term))
                )
        except KeyError:
            return {}

    def _get_terms_emotional_valence(
        self, terms: Sequence[str] | Sequence[Token]
    ) -> dict[str, float]:
        all_emo_weights = collections.defaultdict(list)
        for term in terms:
            emo_weights = self._get_term_emotional_valence(term)
            for emo, weight in emo_weights.items():
                all_emo_weights[emo].append(weight)
        return {
            emo: statistics.mean(weights) for emo, weights in all_emo_weights.items()
        }
