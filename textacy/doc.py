# -*- coding: utf-8 -*-
"""
Load, process, iterate, transform, and save text content paired with metadata
-- a document.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import os
import types

from cytoolz import itertoolz
from spacy import attrs
from spacy.language import Language as SpacyLang
from spacy.tokens.doc import Doc as SpacyDoc
from spacy.tokens.span import Span as SpacySpan
from spacy.tokens.token import Token as SpacyToken

from . import cache
from . import compat
from . import constants
from . import extract
from . import io
from . import network
from . import text_utils
from .spacier import utils as spacy_utils


class Doc(object):
    """
    A text document parsed by spaCy and, optionally, paired with key metadata.
    Transform :class:`Doc` into an easily-customized list of terms, a bag-of-words
    or (more general) bag-of-terms, or a semantic network; save and load parsed
    content and metadata to and from disk; index, slice, and iterate through
    tokens and sentences; and more.

    Initialize from a text and (optional) metadata::

        >>> content = '''
        ...     The apparent symmetry between the quark and lepton families of
        ...     the Standard Model (SM) are, at the very least, suggestive of
        ...     a more fundamental relationship between them. In some Beyond the
        ...     Standard Model theories, such interactions are mediated by
        ...     leptoquarks (LQs): hypothetical color-triplet bosons with both
        ...     lepton and baryon number and fractional electric charge.'''
        >>> metadata = {
        ...     'title': 'A Search for 2nd-generation Leptoquarks at √s = 7 TeV',
        ...     'author': 'Burton DeWilde',
        ...     'pub_date': '2012-08-01'}
        >>> doc = textacy.Doc(content, metadata=metadata)
        >>> print(doc)
        Doc(71 tokens; "The apparent symmetry between the quark and lep...")

    Transform into other, common formats::

        >>> doc.to_bag_of_words(lemmatize=False, as_strings=False)
        {205123: 1, 21382: 1, 17929: 1, 175499: 2, 396: 1, 29774: 1, 27472: 1,
         4498: 1, 1814: 1, 1176: 1, 49050: 1, 287836: 1, 1510365: 1, 6239: 2,
         3553: 1, 5607: 1, 4776: 1, 49580: 1, 6701: 1, 12078: 2, 63216: 1,
         6738: 1, 83061: 1, 5243: 1, 1599: 1}
        >>> doc.to_bag_of_terms(ngrams=2, named_entities=True,
        ...                     lemmatize=True, as_strings=True)
        {'apparent symmetry': 1, 'baryon number': 1, 'electric charge': 1,
         'fractional electric': 1, 'fundamental relationship': 1,
         'hypothetical color': 1, 'lepton family': 1, 'model theory': 1,
         'standard model': 2, 'triplet boson': 1}

    Doc as sequence of tokens, emulating spaCy's "sequence API"::

        >>> doc[49]  # spacy.Token
        leptoquarks
        >>> doc[:3]  # spacy.Span
        The apparent symmetry

    Save to and load from disk::

        >>> doc.save('~/Desktop/leptoquarks.pkl')
        >>> doc = textacy.Doc.load('~/Desktop/leptoquarks.pkl')
        >>> print(doc)
        Doc(71 tokens; "The apparent symmetry between the quark and lep...")

    Args:
        content (str or ``spacy.Doc``): Document content as (unicode) text or an
            already-parsed ``spacy.Doc``. If str, content is processed by models
            loaded with a ``spacy.Language`` and assigned to :attr:`spacy_doc`.
        metadata (dict): Dictionary of relevant information about content. This
            can be helpful when identifying and filtering documents, as well as
            when engineering features for model inputs.
        lang (str or ``spacy.Language`` or callable): Language of document content.
            If known, pass a standard 2-letter language code (e.g. "en"), or the
            name of a spacy model for the desired language (e.g. "en_core_web_md"),
            or an already-instantiated ``spacy.Language`` object. If not known,
            pass a function/callable that takes unicode text as input and outputs
            a standard 2-letter language code.

            The given or detected language str is used to instantiate a corresponding
            ``spacy.Language`` with all models loaded by default, and the appropriate
            2-letter lang code is assigned to :attr:`Doc.lang`.

            **Note:** The ``spacy.Language`` object parses ``content``
            (if str) and sets the :attr:`spacy_vocab` and :attr:`spacy_stringstore`
            attributes. See https://spacy.io/usage/models#section-available for
            available spacy models.

    Attributes:
        lang (str): 2-letter code for language of ``Doc``.
        metadata (dict): Dictionary of relevant information about content.
        spacy_doc (``spacy.Doc``): https://spacy.io/api/doc
        spacy_vocab (``spacy.Vocab``): https://spacy.io/api/vocab
        spacy_stringstore (``spacy.StringStore``): https://spacy.io/api/stringstore
    """

    def __init__(self, content, metadata=None, lang=text_utils.detect_language):
        if isinstance(content, compat.unicode_):
            self._init_from_text(content, metadata, lang)
        elif isinstance(content, SpacyDoc):
            self._init_from_spacy_doc(content, metadata, lang)
        else:
            raise ValueError(
                '`Doc` must be initialized with {} content, not "{}"'.format(
                    {compat.unicode_, SpacyDoc}, type(content)
                )
            )

        self._counted_ngrams = set()
        self._counts = collections.Counter()

    def _init_from_text(self, content, metadata, lang):
        """Doc instantiated from text, so must be parsed with a spacy.Language.
        """
        if isinstance(lang, SpacyLang):
            spacy_lang = lang
            langstr = spacy_lang.lang
        elif isinstance(lang, compat.unicode_):
            spacy_lang = cache.load_spacy(lang)
            langstr = spacy_lang.lang
        elif callable(lang):
            langstr = lang(content)
            spacy_lang = cache.load_spacy(langstr)
        else:
            raise TypeError(
                "`lang` must be {}, not {}".format(
                    {compat.unicode_, SpacyLang, types.FunctionType}, type(lang)
                )
            )
        self.spacy_vocab = spacy_lang.vocab
        self.spacy_stringstore = self.spacy_vocab.strings
        self.spacy_doc = spacy_lang(content)
        self.spacy_doc.user_data["textacy"] = {
            "lang": langstr,
            "metadata": metadata or {},
        }

    def _init_from_spacy_doc(self, content, metadata, lang):
        """Doc instantiated from an already-parsed spacy.Doc. Ensure agreement
        between various spacy objects and inputs, and check for existing metadata
        rather than always overwriting it.
        """
        self.spacy_vocab = content.vocab
        self.spacy_stringstore = self.spacy_vocab.strings
        self.spacy_doc = content
        langstr = self.spacy_vocab.lang
        # these checks are probably unnecessary, but in case a user
        # has done something strange, we should complain...
        if isinstance(lang, SpacyLang):
            if self.spacy_vocab is not lang.vocab:
                raise ValueError(
                    "`spacy.Vocab` used to parse `content` must be the same "
                    "as the one associated with the `lang` param"
                )
        elif isinstance(lang, compat.unicode_):
            # a `lang` as str could be a specific spacy model name,
            # e.g. "en_core_web_sm", while `langstr` would only be "en"
            if not lang.startswith(langstr):
                raise ValueError(
                    "lang of spacy model used to parse `content` ({}) "
                    "must be the same as the `lang` param ({})".format(lang, langstr)
                )
        elif callable(lang) is False:
            raise TypeError(
                "`lang` must be {}, not {}".format(
                    {compat.unicode_, SpacyLang, types.FunctionType}, type(lang)
                )
            )
        # textacy metadata could already be assigned to spacy doc; grab it
        # if it's there and user hasn't supplied *new* metadata to overwrite it
        metadata = (
            metadata or content.user_data.get("textacy", {}).get("metadata") or {}
        )
        self.spacy_doc.user_data["textacy"] = {"lang": langstr, "metadata": metadata}

    def __repr__(self):
        snippet = self.text[:50].replace("\n", " ")
        if len(snippet) == 50:
            snippet = snippet[:47] + "..."
        return 'Doc({} tokens; "{}")'.format(len(self.spacy_doc), snippet)

    def __len__(self):
        return self.n_tokens

    def __getitem__(self, index):
        return self.spacy_doc[index]

    def __iter__(self):
        for tok in self.spacy_doc:
            yield tok

    @property
    def metadata(self):
        """:class:`Doc` metadata, stored in ``SpacyDoc.user_data``."""
        return self.spacy_doc.user_data["textacy"]["metadata"]

    @metadata.setter
    def metadata(self, value):
        self.spacy_doc.user_data["textacy"]["metadata"] = value

    @property
    def lang(self):
        """:class:`Doc` language, stored in ``SpacyDoc.user_data``."""
        return self.spacy_doc.user_data["textacy"]["lang"]

    ##########
    # FILEIO #

    def save(self, filepath):
        """
        Save :class:`Doc` content and metadata to disk, as a ``pickle`` file.

        Args:
            filepath (str): Full path to file on disk where document content and
                metadata are to be saved.

        See Also:
            :meth:`Doc.load()`
        """
        io.write_spacy_docs(self.spacy_doc, filepath)

    @classmethod
    def load(cls, filepath):
        """
        Load pickled content and metadata from disk, and initialize a :class:`Doc`.

        Args:
            filepath (str): Full path to file on disk where document content and
                metadata are saved.

        Returns:
            :class:`Doc`

        See Also:
            :meth:`Doc.save()`
        """
        spacy_doc = list(io.read_spacy_docs(filepath))[0]
        return cls(
            spacy_doc,
            lang=spacy_doc.user_data["textacy"]["lang"],
            metadata=spacy_doc.user_data["textacy"]["metadata"],
        )

    ####################
    # BASIC COMPONENTS #

    @property
    def tokens(self):
        """
        Yield the document's tokens, as tokenized by spaCy. Equivalent to
        iterating directly: ``for token in Doc: <do stuff>``
        """
        for tok in self.spacy_doc:
            yield tok

    @property
    def sents(self):
        """Yield the document's sentences, as segmented by spaCy."""
        for sent in self.spacy_doc.sents:
            yield sent

    @property
    def n_tokens(self):
        """The number of tokens in the document -- including punctuation."""
        return len(self.spacy_doc)

    @property
    def n_sents(self):
        """The number of sentences in the document."""
        return sum(1 for _ in self.spacy_doc.sents)

    def merge(self, spans):
        """
        Merge spans *in-place* within :class:`Doc` so that each takes up a single
        token. Note: All cached counts on this doc are cleared after a merge.

        Args:
            spans (Iterable[``spacy.Span``]): for example, the results from
                :func:`extract.named_entities() <textacy.extract.named_entities>`
                or :func:`extract.pos_regex_matches() <textacy.extract.pos_regex_matches>`
        """
        spacy_utils.merge_spans(spans)
        # reset counts, since merging spans invalidates existing counts
        self._counts.clear()
        self._counted_ngrams = set()

    def count(self, term):
        """
        Get the number of occurrences (i.e. count) of ``term`` in :class:`Doc`.

        Args:
            term (str or int or ``spacy.Token`` or ``spacy.Span``): The term to
                be counted can be given as a string, a unique integer id, a
                spacy token, or a spacy span. Counts for the same term given in
                different forms are the same!

        Returns:
            int: Count of ``term`` in :class:`Doc`.

        .. tip:: Counts are cached. The first time a single word's count is
           looked up, *all* words' counts are saved, resulting in a slower
           runtime the first time but orders of magnitude faster runtime for
           subsequent calls for this or any other word. Similarly, if a
           bigram's count is looked up, all bigrams' counts are stored — etc.
           If spans are merged using :meth:`Doc.merge()`, all cached counts are
           deleted, since merging spans will invalidate many counts. Better to
           merge first, count second!
        """
        # figure out what object we're dealing with here; convert as necessary
        if isinstance(term, compat.unicode_):
            term_text = term
            term_id = self.spacy_stringstore.add(term_text)
            term_len = term_text.count(" ") + 1
        elif isinstance(term, int):
            term_id = term
            term_text = self.spacy_stringstore[term_id]
            term_len = term_text.count(" ") + 1
        elif isinstance(term, SpacyToken):
            term_text = term.text
            term_id = self.spacy_stringstore.add(term_text)
            term_len = 1
        elif isinstance(term, SpacySpan):
            term_text = term.text
            term_id = self.spacy_stringstore.add(term_text)
            term_len = len(term)
        # we haven't counted terms of this length; let's do that now
        if term_len not in self._counted_ngrams:
            if term_len == 1:
                self._counts += collections.Counter(
                    word.orth
                    for word in extract.words(
                        self, filter_stops=False, filter_punct=False, filter_nums=False
                    )
                )
            else:
                self._counts += collections.Counter(
                    self.spacy_stringstore.add(ngram.text)
                    for ngram in extract.ngrams(
                        self,
                        term_len,
                        filter_stops=False,
                        filter_punct=False,
                        filter_nums=False,
                    )
                )
            self._counted_ngrams.add(term_len)
        return self._counts[term_id]

    ###############
    # DOC AS TEXT #

    @property
    def text(self):
        """Return the document's raw text."""
        return self.spacy_doc.text_with_ws

    @property
    def tokenized_text(self):
        """Return text as an ordered, nested list of tokens per sentence."""
        return [[token.text for token in sent] for sent in self.spacy_doc.sents]

    @property
    def pos_tagged_text(self):
        """Return text as an ordered, nested list of (token, POS) pairs per sentence."""
        return [
            [(token.text, token.pos_) for token in sent]
            for sent in self.spacy_doc.sents
        ]

    #################
    # TRANSFORM DOC #

    def to_terms_list(
        self,
        ngrams=(1, 2, 3),
        named_entities=True,
        normalize="lemma",
        as_strings=False,
        **kwargs
    ):
        """
        Transform :class:`Doc` into a sequence of ngrams and/or named entities, which
        aren't necessarily in order of appearance, where each term appears in
        the list with the same frequency that it appears in :class:`Doc`.

        Args:
            ngrams (int or Set[int]): n of which n-grams to include; ``(1, 2, 3)``
                (default) includes unigrams (words), bigrams, and trigrams; `2`
                if only bigrams are wanted; falsy (e.g. False) to not include any
            named_entities (bool): if True (default), include named entities
                in the terms list; note: if ngrams are also included, named
                entities are added *first*, and any ngrams that exactly overlap
                with an entity are skipped to prevent double-counting
            normalize (str or callable): if 'lemma', lemmatize terms; if 'lower',
                lowercase terms; if false-y, use the form of terms as they appear
                in doc; if a callable, must accept a ``spacy.Token`` or ``spacy.Span``
                and return a str, e.g. :func:`textacy.spacier.utils.get_normalized_text()`
            as_strings (bool): if True, terms are returned as strings; if False
                (default), terms are returned as their unique integer ids
            kwargs:
                - filter_stops (bool)
                - filter_punct (bool)
                - filter_nums (bool)
                - include_pos (str or Set[str])
                - exclude_pos (str or Set[str])
                - min_freq (int)
                - include_types (str or Set[str])
                - exclude_types (str or Set[str]
                - drop_determiners (bool)

                see :func:`extract.words <textacy.extract.words>`,
                :func:`extract.ngrams <textacy.extract.ngrams>`,
                and :func:`extract.named_entities <textacy.extract.named_entities>`
                for more information on these parameters

        Yields:
            int or str: the next term in the terms list, either as a unique
            integer id or as a string

        Raises:
            ValueError: if neither ``named_entities`` nor ``ngrams`` are included

        Note:
            Despite the name, this is a generator function; to get an
            actual list of terms, call ``list(doc.to_terms_list())``.
        """
        if not named_entities and not ngrams:
            raise ValueError("either `named_entities` or `ngrams` must be included")
        if ngrams and isinstance(ngrams, int):
            ngrams = (ngrams,)
        if named_entities is True:
            ne_kwargs = {
                "include_types": kwargs.get("include_types"),
                "exclude_types": kwargs.get("exclude_types"),
                "drop_determiners": kwargs.get("drop_determiners", True),
                "min_freq": kwargs.get("min_freq", 1),
            }
            # if numeric ngrams are to be filtered, we should filter numeric entities
            if ngrams and kwargs.get("filter_nums") is True:
                if ne_kwargs["exclude_types"]:
                    if isinstance(
                        ne_kwargs["exclude_types"], (set, frozenset, list, tuple)
                    ):
                        ne_kwargs["exclude_types"] = set(ne_kwargs["exclude_types"])
                        ne_kwargs["exclude_types"].add(constants.NUMERIC_NE_TYPES)
                else:
                    ne_kwargs["exclude_types"] = constants.NUMERIC_NE_TYPES
        if ngrams:
            ngram_kwargs = {
                "filter_stops": kwargs.get("filter_stops", True),
                "filter_punct": kwargs.get("filter_punct", True),
                "filter_nums": kwargs.get("filter_nums", False),
                "include_pos": kwargs.get("include_pos"),
                "exclude_pos": kwargs.get("exclude_pos"),
                "min_freq": kwargs.get("min_freq", 1),
            }
            # if numeric entities are to be filtered, we should filter numeric ngrams
            if (
                named_entities
                and ne_kwargs["exclude_types"]
                and any(
                    ne_type in ne_kwargs["exclude_types"]
                    for ne_type in constants.NUMERIC_NE_TYPES
                )
            ):
                ngram_kwargs["filter_nums"] = True

        terms = []
        # special case: ensure that named entities aren't double-counted when
        # adding words or ngrams that were already added as named entities
        if named_entities is True and ngrams:
            ents = tuple(extract.named_entities(self, **ne_kwargs))
            ent_idxs = {(ent.start, ent.end) for ent in ents}
            terms.append(ents)
            for n in ngrams:
                if n == 1:
                    terms.append(
                        (
                            word
                            for word in extract.words(self, **ngram_kwargs)
                            if (word.i, word.i + 1) not in ent_idxs
                        )
                    )
                else:
                    terms.append(
                        (
                            ngram
                            for ngram in extract.ngrams(self, n, **ngram_kwargs)
                            if (ngram.start, ngram.end) not in ent_idxs
                        )
                    )
        # otherwise, no need to check for overlaps
        else:
            if named_entities is True:
                terms.append(extract.named_entities(self, **ne_kwargs))
            else:
                for n in ngrams:
                    if n == 1:
                        terms.append(extract.words(self, **ngram_kwargs))
                    else:
                        terms.append(extract.ngrams(self, n, **ngram_kwargs))

        terms = itertoolz.concat(terms)

        # convert token and span objects into integer ids
        if as_strings is False:
            if normalize == "lemma":
                for term in terms:
                    try:
                        yield term.lemma
                    except AttributeError:
                        yield self.spacy_stringstore.add(term.lemma_)
            elif normalize == "lower":
                for term in terms:
                    try:
                        yield term.lower
                    except AttributeError:
                        yield self.spacy_stringstore.add(term.lower_)
            elif not normalize:
                for term in terms:
                    try:
                        yield term.orth
                    except AttributeError:
                        yield self.spacy_stringstore.add(term.text)
            else:
                for term in terms:
                    yield self.spacy_stringstore.add(normalize(term))

        # convert token and span objects into strings
        else:
            if normalize == "lemma":
                for term in terms:
                    yield term.lemma_
            elif normalize == "lower":
                for term in terms:
                    yield term.lower_
            elif not normalize:
                for term in terms:
                    yield term.text
            else:
                for term in terms:
                    yield normalize(term)

    def to_bag_of_words(self, normalize="lemma", weighting="count", as_strings=False):
        """
        Transform :class:`Doc` into a bag-of-words: the set of unique words in
        :class:`Doc` mapped to their absolute, relative, or binary frequency of
        occurrence.

        Args:
            normalize (str): if 'lemma', lemmatize words before counting; if
                'lower', lowercase words before counting; otherwise, words are
                counted using the form with which they they appear in doc
            weighting ({'count', 'freq', 'binary'}): Type of weight to assign to
                words. If 'count' (default), weights are the absolute number of
                occurrences (count) of word in doc. If 'binary', all counts are
                set equal to 1. If 'freq', word counts are normalized by the
                total token count, giving their relative frequency of occurrence.
                Note: The resulting set of frequencies won't (necessarily) sum
                to 1.0, since punctuation and stop words are filtered out after
                counts are normalized.
            as_strings (bool): if True, words are returned as strings; if False
                (default), words are returned as their unique integer ids

        Returns:
            dict: mapping of a unique word id or string (depending on the value
            of ``as_strings``) to its absolute, relative, or binary frequency
            of occurrence (depending on the value of ``weighting``).
        """
        if weighting not in {"count", "freq", "binary"}:
            raise ValueError('weighting "{}" is invalid'.format(weighting))
        count_by = (
            attrs.LEMMA
            if normalize == "lemma"
            else attrs.LOWER
            if normalize == "lower"
            else attrs.ORTH
        )
        word_to_weight = self.spacy_doc.count_by(count_by)
        if weighting == "freq":
            n_tokens = self.n_tokens
            word_to_weight = {
                id_: weight / n_tokens for id_, weight in word_to_weight.items()
            }
        elif weighting == "binary":
            word_to_weight = {word: 1 for word in word_to_weight.keys()}

        bow = {}
        if as_strings is False:
            for id_, count in word_to_weight.items():
                lexeme = self.spacy_vocab[id_]
                if lexeme.is_stop or lexeme.is_punct or lexeme.is_space:
                    continue
                bow[id_] = count
        else:
            for id_, count in word_to_weight.items():
                lexeme = self.spacy_vocab[id_]
                if lexeme.is_stop or lexeme.is_punct or lexeme.is_space:
                    continue
                bow[self.spacy_stringstore[id_]] = count
        return bow

    def to_bag_of_terms(
        self,
        ngrams=(1, 2, 3),
        named_entities=True,
        normalize="lemma",
        weighting="count",
        as_strings=False,
        **kwargs
    ):
        """
        Transform :class:`Doc` into a bag-of-terms: the set of unique terms in
        :class:`Doc` mapped to their frequency of occurrence, where "terms"
        includes ngrams and/or named entities.

        Args:
            ngrams (int or Set[int]): n of which n-grams to include; ``(1, 2, 3)``
                (default) includes unigrams (words), bigrams, and trigrams; `2`
                if only bigrams are wanted; falsy (e.g. False) to not include any
            named_entities (bool): if True (default), include named entities;
                note: if ngrams are also included, any ngrams that exactly
                overlap with an entity are skipped to prevent double-counting
            normalize (str or callable): if 'lemma', lemmatize terms; if 'lower',
                lowercase terms; if false-y, use the form of terms as they appear
                in doc; if a callable, must accept a ``spacy.Token`` or ``spacy.Span``
                and return a str, e.g. :func:`textacy.spacier.utils.get_normalized_text()`
            weighting ({'count', 'freq', 'binary'}): Type of weight to assign to
                terms. If 'count' (default), weights are the absolute number of
                occurrences (count) of term in doc. If 'binary', all counts are
                set equal to 1. If 'freq', term counts are normalized by the
                total token count, giving their relative frequency of occurrence.
            as_strings (bool): if True, words are returned as strings; if False
                (default), words are returned as their unique integer ids
            kwargs:
                - filter_stops (bool)
                - filter_punct (bool)
                - filter_nums (bool)
                - include_pos (str or Set[str])
                - exclude_pos (str or Set[str])
                - min_freq (int)
                - include_types (str or Set[str])
                - exclude_types (str or Set[str]
                - drop_determiners (bool)

                See :func:`extract.words() <textacy.extract.words>`,
                :func:`extract.ngrams() <textacy.extract.ngrams>`,
                and :func:`extract.named_entities() <textacy.extract.named_entities>`
                for more information on these parameters.

        Returns:
            dict: mapping of a unique term id or string (depending on the value
            of ``as_strings``) to its absolute, relative, or binary frequency
            of occurrence (depending on the value of ``weighting``).

        See Also:
            :meth:`Doc.to_terms_list() <Doc.to_terms_list>`
        """
        if weighting not in {"count", "freq", "binary"}:
            raise ValueError('weighting "{}" is invalid'.format(weighting))
        terms_list = self.to_terms_list(
            ngrams=ngrams,
            named_entities=named_entities,
            normalize=normalize,
            as_strings=as_strings,
            **kwargs
        )
        bot = itertoolz.frequencies(terms_list)
        if weighting == "freq":
            n_tokens = self.n_tokens
            bot = {term: weight / n_tokens for term, weight in bot.items()}
        elif weighting == "binary":
            bot = {term: 1 for term in bot.keys()}
        return bot

    def to_semantic_network(
        self,
        nodes="words",
        normalize="lemma",
        edge_weighting="default",
        window_width=10,
    ):
        """
        Transform :class:`Doc` into a semantic network, where nodes are either
        'words' or 'sents' and edges between nodes may be weighted in different ways.

        Args:
            nodes ({'words', 'sents'}): type of doc component to use as nodes
                in the semantic network
            normalize (str or callable): if 'lemma', lemmatize terms; if 'lower',
                lowercase terms; if false-y, use the form of terms as they appear
                in doc; if a callable, must accept a ``spacy.Token`` or ``spacy.Span``
                (if ``nodes`` = 'words' or 'sents', respectively) and return a
                str, e.g. :func:`textacy.spacier.utils.get_normalized_text()`
            edge_weighting (str): type of weighting to apply to edges
                between nodes; if ``nodes == 'words'``, options are {'cooc_freq', 'binary'},
                if ``nodes == 'sents'``, options are {'cosine', 'jaccard'}; if
                'default', 'cooc_freq' or 'cosine' will be automatically used
            window_width (int): size of sliding window over terms that
                determines which are said to co-occur; only applicable if 'words'

        Returns:
            ``networkx.Graph``: where nodes represent either terms or sentences
            in doc; edges, the relationships between them.

        Raises:
            ValueError: if ``nodes`` is neither 'words' nor 'sents'.

        See Also:
            - :func:`terms_to_semantic_network() <textacy.network.terms_to_semantic_network>`
            - :func:`sents_to_semantic_network() <textacy.network.sents_to_semantic_network>`
        """
        if nodes == "words":
            if edge_weighting == "default":
                edge_weighting = "cooc_freq"
            return network.terms_to_semantic_network(
                list(extract.words(self)),
                normalize=normalize,
                window_width=window_width,
                edge_weighting=edge_weighting,
            )
        elif nodes == "sents":
            if edge_weighting == "default":
                edge_weighting = "cosine"
            return network.sents_to_semantic_network(
                list(self.sents), normalize=normalize, edge_weighting=edge_weighting
            )
        else:
            msg = 'nodes "{}" not valid; must be in {}'.format(
                nodes, {"words", "sents"}
            )
            raise ValueError(msg)
