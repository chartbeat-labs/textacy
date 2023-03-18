# mypy: ignore-errors
"""
ConceptNet
----------

ConceptNet is a multilingual knowledge base, representing common words and phrases
and the common-sense relationships between them. This information is collected from
a variety of sources, including crowd-sourced resources (e.g. Wiktionary, Open Mind
Common Sense), games with a purpose (e.g. Verbosity, nadya.jp), and expert-created
resources (e.g. WordNet, JMDict).

The interface in textacy gives access to several key relationships between terms
that are useful in a variety of NLP tasks:

    - antonyms: terms that are opposites of each other in some relevant way
    - hyponyms: terms that are subtypes or specific instances of other terms
    - meronyms: terms that are parts of other terms
    - synonyms: terms that are sufficiently similar that they may be used interchangeably
"""
from __future__ import annotations

import collections
import logging
from typing import ClassVar, Optional

from spacy.tokens import Span, Token
from tqdm import tqdm

from .. import constants
from .. import io as tio
from .. import types, utils
from .base import Resource


LOGGER = logging.getLogger(__name__)

NAME = "concept_net"
META = {
    "site_url": "http://conceptnet.io",
    "publication_url": "https://arxiv.org/abs/1612.03975",
    "description": (
        "An open, multilingual semantic network of general knowledge, "
        "designed to help computers understand the meanings of words."
    ),
}
DOWNLOAD_ROOT = "https://s3.amazonaws.com/conceptnet/downloads/{year}/edges/conceptnet-assertions-{version}.csv.gz"


class ConceptNet(Resource):
    """
    Interface to ConceptNet, a multilingual knowledge base representing common words
    and phrases and the common-sense relationships between them.

    Download the data (one time only!), and save its contents to disk::

        >>> import textacy.resources
        >>> rs = textacy.resources.ConceptNet()
        >>> rs.download()
        >>> rs.info
        {'name': 'concept_net',
         'site_url': 'http://conceptnet.io',
         'publication_url': 'https://arxiv.org/abs/1612.03975',
         'description': 'An open, multilingual semantic network of general knowledge, designed to help computers understand the meanings of words.'}

    Access other same-language terms related to a given term in a variety of ways::

        >>> rs.get_synonyms("spouse", lang="en", sense="n")
        ['mate', 'married person', 'better half', 'partner']
        >>> rs.get_antonyms("love", lang="en", sense="v")
        ['detest', 'hate', 'loathe']
        >>> rs.get_hyponyms("marriage", lang="en", sense="n")
        ['cohabitation situation', 'union', 'legal agreement', 'ritual', 'family', 'marital status']

    **Note:** The very first time a given relationship is accessed, the full ConceptNet db
    must be parsed and split for fast future access. This can take a couple minutes;
    be patient.

    When passing a spaCy ``Token`` or ``Span``, the corresponding ``lang`` and ``sense``
    are inferred automatically from the object::

        >>> text = "The quick brown fox jumps over the lazy dog."
        >>> doc = textacy.make_spacy_doc(text, lang="en")
        >>> rs.get_synonyms(doc[1])  # quick
        ['flying', 'fast', 'rapid', 'ready', 'straightaway', 'nimble', 'speedy', 'warm']
        >>> rs.get_synonyms(doc[4:5])  # jumps over
        ['leap', 'startle', 'hump', 'flinch', 'jump off', 'skydive', 'jumpstart', ...]

    Many terms won't have entries, for actual linguistic reasons or because the db's
    coverage of a given language's vocabulary isn't comprehensive::

        >>> rs.get_meronyms(doc[3])  # fox
        []
        >>> rs.get_antonyms(doc[7])  # lazy
        []

    Args:
        data_dir: Path to directory on disk under which resource data is stored,
            i.e. ``/path/to/data_dir/concept_net``.
        version ({"5.7.0", "5.6.0", "5.5.5"}): Version string of the ConceptNet db
            to use. Since newer versions typically represent improvements over earlier
            versions, you'll probably want "5.7.0" (the default value).
    """

    _version_years: ClassVar[dict[str, int]] = {
        "5.7.0": 2019,
        "5.6.0": 2018,
        "5.5.5": 2017,
    }
    _pos_map: ClassVar[dict[str, str]] = {
        "NOUN": "n",
        "VERB": "v",
        "ADJ": "a",
        "ADV": "r",
    }

    def __init__(
        self,
        data_dir: types.PathLike = constants.DEFAULT_DATA_DIR.joinpath(NAME),
        version: str = "5.7.0",
    ):
        super().__init__(NAME, meta=META)
        self.version = version
        self.data_dir = utils.to_path(data_dir).resolve().joinpath(self.version)
        self._filename = "conceptnet-assertions-{}.csv.gz".format(self.version)
        self._filepath = self.data_dir.joinpath(self._filename)
        self._antonyms = None
        self._hyponyms = None
        self._meronyms = None
        self._synonyms = None

    def download(self, *, force: bool = False):
        """
        Download resource data as a gzipped csv file,
        then save it to disk under the :attr:`ConceptNet.data_dir` directory.

        Args:
            force (bool): If True, download resource data, even if it already
                exists on disk; otherwise, don't re-download the data.
        """
        url = DOWNLOAD_ROOT.format(
            version=self.version, year=self._version_years[self.version]
        )
        tio.download_file(
            url,
            filename=self._filename,
            dirpath=self.data_dir,
            force=force,
        )

    @property
    def filepath(self) -> Optional[str]:
        """
        str: Full path on disk for the ConceptNet gzipped csv file
        corresponding to the given :attr:`ConceptNet.data_dir`.
        """
        if self._filepath.is_file():
            return str(self._filepath)
        else:
            return None

    def _get_relation_data(
        self, relation: str, is_symmetric: bool = False
    ) -> dict[str, dict[str, dict[str, list[str]]]]:
        if not self.filepath:
            raise OSError(
                "resource file {} not found;\n"
                "has the data been downloaded yet?".format(self._filepath)
            )
        rel_fname = "{}.json.gz".format(_split_uri(relation)[1].lower())
        rel_fpath = self.data_dir.joinpath(rel_fname)
        if rel_fpath.is_file():
            LOGGER.debug("loading data for '%s' relation from %s", relation, rel_fpath)
            return next(
                tio.read_json(rel_fpath, mode="rt", encoding="utf-8", lines=False)
            )
        else:
            rel_data = collections.defaultdict(
                lambda: collections.defaultdict(lambda: collections.defaultdict(set))
            )
            LOGGER.info(
                "preparing data for '%s' relation; this may take a while...", relation
            )
            rows = tio.read_csv(
                self.filepath, encoding="utf-8", delimiter="\t", quoting=1
            )
            with tqdm() as pbar:
                for row in rows:
                    pbar.update(1)
                    _, rel_type, start_uri, end_uri, _ = row
                    if rel_type < relation:
                        continue
                    elif rel_type > relation:
                        break
                    start_lang, start_term, start_sense = _parse_concept_uri(start_uri)
                    end_lang, end_term, end_sense = _parse_concept_uri(end_uri)
                    if start_lang == end_lang and start_term != end_term:
                        rel_data[start_lang][start_term][start_sense].add(end_term)
                        if is_symmetric:
                            rel_data[start_lang][end_term][end_sense].add(start_term)
            # make relation data json-able (i.e. cast set => list)
            for terms in rel_data.values():
                for senses in terms.values():
                    for sense, rel_terms in senses.items():
                        senses[sense] = list(rel_terms)
            LOGGER.info("saving data for '%s' relation to %s", relation, rel_fpath)
            tio.write_json(rel_data, rel_fpath, mode="wt", encoding="utf-8")
            return rel_data

    def _get_relation_values(
        self,
        rel_data,
        term: str | types.SpanLike,
        lang: Optional[str] = None,
        sense: Optional[str] = None,
    ) -> list[str]:
        if lang is not None and lang not in rel_data:
            raise ValueError(
                "lang='{}' is invalid; valid langs are {}".format(
                    lang, sorted(rel_data.keys())
                )
            )
        if sense is not None:
            # let's be kind and automatically convert standard POS strings
            if sense in self._pos_map.keys():
                sense = self._pos_map[sense]
            # otherwise, return no results as is done when auto-inferrence of sense
            # results in an invalid value
            elif sense not in self._pos_map.values():
                return []
        if isinstance(term, str):
            if not (lang and sense):
                raise ValueError(
                    "if `term` is a string, both `lang` and `sense` must be specified"
                )
            else:
                norm_terms = [term.replace(" ", "_").lower()]
        elif isinstance(term, (Span, Token)):
            norm_terms = [
                term.text.replace(" ", "_").lower(),
                term.lemma_.replace(" ", "_").lower(),
            ]
            if not lang:
                try:
                    lang = term.lang_  # token
                except AttributeError:
                    lang = term[0].lang_  # span
            if not sense:
                try:
                    sense = self._pos_map[term.pos_]  # token
                except AttributeError:
                    sense = self._pos_map[term[0].pos_]  # span
                except KeyError:
                    return []
        else:
            raise TypeError(
                "`term` must be one of {}, not {}".format(
                    {str, Span, Token}, type(term)
                )
            )
        # TODO: implement an out-of-vocabulary strategy? for example,
        # https://github.com/commonsense/conceptnet-numberbatch#out-of-vocabulary-strategy
        for norm_term in norm_terms:
            try:
                return rel_data[lang][norm_term].get(sense, [])
            except KeyError:
                pass
        return []

    @property
    def antonyms(self) -> dict[str, dict[str, dict[str, list[str]]]]:
        """
        Mapping of language code to term to sense to set of term's antonyms --
        opposites of the term in some relevant way, like being at opposite ends
        of a scale or fundamentally similar but with a key difference between them --
        such as black <=> white or hot <=> cold. Note that this relationship is symmetric.

        Based on the "/r/Antonym" relation in ConceptNet.
        """
        if not self._antonyms:
            self._antonyms = self._get_relation_data("/r/Antonym", is_symmetric=True)
        return self._antonyms

    def get_antonyms(
        self,
        term: str | types.SpanLike,
        *,
        lang: Optional[str] = None,
        sense: Optional[str] = None,
    ) -> list[str]:
        """
        Args:
            term
            lang: Standard code for the language of ``term``.
            sense: Sense in which ``term`` is used in context, which in practice
                is just its part of speech. Valid values: "n" or "NOUN", "v" or "VERB",
                "a" or "ADJ", "r" or "ADV".
        """
        return self._get_relation_values(self.antonyms, term, lang=lang, sense=sense)

    @property
    def hyponyms(self) -> dict[str, dict[str, dict[str, list[str]]]]:
        """
        Mapping of language code to term to sense to set of term's hyponyms --
        subtypes or specific instances of the term --
        such as car => vehicle or Chicago => city. Every A is a B.

        Based on the "/r/IsA" relation in ConceptNet.
        """
        if not self._hyponyms:
            self._hyponyms = self._get_relation_data("/r/IsA", is_symmetric=False)
        return self._hyponyms

    def get_hyponyms(
        self,
        term: str | types.SpanLike,
        *,
        lang: Optional[str] = None,
        sense: Optional[str] = None,
    ) -> list[str]:
        """
        Args:
            term
            lang: Standard code for the language of ``term``.
            sense: Sense in which ``term`` is used in context, which in practice
                is just its part of speech. Valid values: "n" or "NOUN", "v" or "VERB",
                "a" or "ADJ", "r" or "ADV".
        """
        return self._get_relation_values(self.hyponyms, term, lang=lang, sense=sense)

    @property
    def meronyms(self) -> dict[str, dict[str, dict[str, list[str]]]]:
        """
        Mapping of language code to term to sense to set of term's meronyms --
        parts of the term -- such as gearshift => car.

        Based on the "/r/PartOf" relation in ConceptNet.
        """
        if not self._meronyms:
            self._meronyms = self._get_relation_data("/r/PartOf", is_symmetric=False)
        return self._meronyms

    def get_meronyms(
        self,
        term: str | types.SpanLike,
        *,
        lang: Optional[str] = None,
        sense: Optional[str] = None,
    ) -> list[str]:
        """
        Args:
            term
            lang: Standard code for the language of ``term``.
            sense: Sense in which ``term`` is used in context, which in practice
                is just its part of speech. Valid values: "n" or "NOUN", "v" or "VERB",
                "a" or "ADJ", "r" or "ADV".

        Returns:
            list[str]
        """
        return self._get_relation_values(self.meronyms, term, lang=lang, sense=sense)

    @property
    def synonyms(self) -> dict[str, dict[str, dict[str, list[str]]]]:
        """
        Mapping of language code to term to sense to set of term's synonyms --
        sufficiently similar concepts that they may be used interchangeably --
        such as sunlight <=> sunshine. Note that this relationship is symmetric.

        Based on the "/r/Synonym" relation in ConceptNet.
        """
        if not self._synonyms:
            self._synonyms = self._get_relation_data("/r/Synonym", is_symmetric=True)
        return self._synonyms

    def get_synonyms(
        self,
        term: str | types.SpanLike,
        *,
        lang: Optional[str] = None,
        sense: Optional[str] = None,
    ) -> list[str]:
        """
        Args:
            term
            lang: Standard code for the language of ``term``.
            sense: Sense in which ``term`` is used in context, which in practice
                is just its part of speech. Valid values: "n" or "NOUN", "v" or "VERB",
                "a" or "ADJ", "r" or "ADV".
        """
        return self._get_relation_values(self.synonyms, term, lang=lang, sense=sense)


def _split_uri(uri: str) -> list[str]:
    """Get slash-delimited parts of a ConceptNet URI."""
    uri = uri.lstrip("/")
    if not uri:
        return []
    return uri.split("/")


def _parse_concept_uri(uri: str) -> tuple[str, str, str]:
    """Extract language, term, and sense from a ConceptNet "concept" URI."""
    if not uri.startswith("/c/"):
        raise ValueError(f"invalid concept uri: {uri}")
    uri = _split_uri(uri)
    if len(uri) == 3:
        _, lang, term = uri
        sense = None
    elif len(uri) == 4:
        _, lang, term, sense = uri
    elif len(uri) > 4:
        _, lang, term, sense, *_ = uri
    elif len(uri) < 3:
        raise ValueError(f"not enough parts in uri: {uri}")
    term = term.replace("_", " ")
    return lang, term, sense
