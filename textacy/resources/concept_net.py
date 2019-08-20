import collections
import json
import logging

from spacy.parts_of_speech import ADJ, ADV, NOUN, VERB
from spacy.tokens import Span, Token
from tqdm import tqdm

from .. import constants, io, utils
from ..datasets import utils as ds_utils  # TODO: move this functionality into io.utils?
from .base import Resource


LOGGER = logging.getLogger(__name__)

NAME = "concept_net"
META = {
    "site_url": "http://conceptnet.io",
    "publication_url": "https://arxiv.org/abs/1612.03975",
    "description": (
        "An open, multilingual semantic network of general knowledge, "
        "designed to help computers understand the meanings of words.",
    )
}
DOWNLOAD_ROOT = "https://s3.amazonaws.com/conceptnet/downloads/{year}/edges/conceptnet-assertions-{version}.csv.gz"


class ConceptNet(Resource):
    """
    Args:
        data_dir (str or :class:`pathlib.Path`)
        version ({"5.7.0", "5.6.0", "5.5.5"})
    """

    _version_years = {"5.7.0": 2019, "5.6.0": 2018, "5.5.5": 2017}
    _pos_map = {NOUN: "n", VERB: "v", ADJ: "a", ADV: "r"}

    def __init__(
        self,
        data_dir=constants.DEFAULT_DATA_DIR.joinpath(NAME),
        version="5.7.0",
    ):
        super().__init__(NAME, meta=META)
        self.version = version
        self.data_dir = utils.to_path(data_dir).resolve().joinpath(self.version)
        self._filename = "conceptnet-assertions-{}.csv.gz".format(self.version)
        self._filepath = self.data_dir.joinpath(self._filename)
        self._antonyms = None
        self._synonyms = None

    def download(self, *, force=False):
        """
        Download resource data as a gzipped csv file,
        then save it to disk under the :attr:`ConceptNet.data_dir` directory.

        Args:
            force (bool): If True, download resource data, even if it already
                exists on disk; otherwise, don't re-download the data.
        """
        url = DOWNLOAD_ROOT.format(
            version=self.version, year=self._version_years[self.version])
        ds_utils.download_file(
            url,
            filename=self._filename,
            dirpath=self.data_dir,
            force=force,
        )

    @property
    def filepath(self):
        """
        str: Full path on disk for the ConceptNet gzipped csv file
        corresponding to the given :attr:`ConceptNet.data_dir`.
        """
        if self._filepath.is_file():
            return str(self._filepath)
        else:
            return None

    def _get_relation_data(self, relation, is_symmetric=False):
        if not self.filepath:
            raise OSError(
                "resource file {} not found;\n"
                "has the data been downloaded yet?".format(self._filepath)
            )
        rel_fname = "{}.json".format(_split_uri(relation)[1].lower())
        rel_fpath = self.data_dir.joinpath(rel_fname)
        if rel_fpath.is_file():
            LOGGER.debug("loading data for '%s' relation from %s", relation, rel_fpath)
            return next(
                io.read_json(rel_fpath, mode="rt", encoding="utf-8", lines=False)
            )
        else:
            rel_data = collections.defaultdict(
                lambda: collections.defaultdict(
                    lambda: collections.defaultdict(set)
                )
            )
            LOGGER.info(
                "preparing data for '%s' relation; this may take a while...", relation)
            rows = io.read_csv(self.filepath, delimiter="\t", quoting=1)
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
                    # TODO: determine if requiring same sense is too string, i.e.
                    # start_sense == end_sense ?
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
            io.write_json(rel_data, rel_fpath, mode="wt", encoding="utf-8")
            return rel_data

    def _get_relation_values(self, rel_data, term, lang=None, sense=None):
        """
        Args:
            term (str or :class:`spacy.tokens.Token` or :class:`spacy.tokens.Span`)
            lang (str)
            sense (str)

        Returns:
            List[str]
        """
        if isinstance(term, str):
            if not (lang and sense):
                raise ValueError(
                    "if `term` is a string, both `lang` and `sense` must be specified")
            else:
                norm_terms = [term.replace(" ", "_").lower()]
        elif isinstance(term, (Span, Token)):
            norm_terms = [
                term.text.replace(" ", "_").lower(),
                term.lemma_.replace(" ", "_").lower(),
            ]
            lang = term.lang_
            try:
                sense = self._pos_map[term.pos]
            except KeyError:
                return []
        else:
            raise TypeError(
                "`term` must be one of {}, not {}".format(
                    {str, Span, Token}, type(term)
                )
            )
        if lang not in rel_data:
            raise ValueError(
                "lang='{}' is invalid; available langs are {}".format(
                    lang, sorted(rel_data.keys())
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
    def antonyms(self):
        """
        Dict[str, Dict[str, Dict[str, List[str]]]]: Mapping of language code to term to
        sense to set of term's antonyms. Careful, this is a _large_ nested dictionary.
        """
        if not self._antonyms:
            self._antonyms = self._get_relation_data("/r/Antonym", is_symmetric=True)
        return self._antonyms

    def get_antonyms(self, term, *, lang=None, sense=None):
        """
        Args:
            term (str or :class:`spacy.tokens.Token` or :class:`spacy.tokens.Span`)
            lang (str)
            sense (str)

        Returns:
            List[str]
        """
        return self._get_relation_values(self.antonyms, term, lang=lang, sense=sense)

    @property
    def synonyms(self):
        """
        Dict[str, Dict[str, Dict[str, List[str]]]]: Mapping of language code to term to
        sense to set of term's synonyms. Careful, this is a _large_ nested dictionary.
        """
        if not self._synonyms:
            self._synonyms = self._get_relation_data("/r/Synonym", is_symmetric=True)
        return self._synonyms

    def get_synonyms(self, term, *, lang=None, sense=None):
        """
        Args:
            term (str or :class:`spacy.tokens.Token` or :class:`spacy.tokens.Span`)
            lang (str)
            sense (str)

        Returns:
            List[str]
        """
        return self._get_relation_values(self.synonyms, term, lang=lang, sense=sense)


def _split_uri(uri):
    """
    Get slash-delimited parts of a ConceptNet URI.

    Args:
        uri (str)

    Returns:
        List[str]
    """
    uri = uri.lstrip("/")
    if not uri:
        return []
    return uri.split("/")


def _parse_concept_uri(uri):
    """
    Extract language, term, and sense from a ConceptNet "concept" URI.

    Args:
        uri (str)

    Returns:
        Tuple[str, str, str]: Language, term, sense.
    """
    if not uri.startswith("/c/"):
        raise ValueError("invalid concept uri: {}".format(uri))
    uri = _split_uri(uri)
    if len(uri) == 3:
        _, lang, term = uri
        sense = None
    elif len(uri) == 4:
        _, lang, term, sense = uri
    elif len(uri) > 4:
        _, lang, term, sense, *_ = uri
    elif len(uri) < 3:
        raise ValueError("not enough parts in uri: {}".format(uri))
    term = term.replace("_", " ")
    return lang, term, sense
