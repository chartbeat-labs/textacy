# -*- coding: utf-8 -*-
"""
Load, process, iterate, transform, and save a collection of documents — a corpus.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import copy
import logging
import math
import multiprocessing
import os

import numpy as np
from cytoolz import itertoolz
from spacy.language import Language as SpacyLang
from spacy.pipeline import DependencyParser
from spacy.tokens.doc import Doc as SpacyDoc
from spacy.util import get_lang_class

from . import cache
from . import compat
from . import io
from . import utils
from .doc import Doc

LOGGER = logging.getLogger(__name__)


class Corpus(object):
    """
    An ordered collection of :class:`Doc <textacy.doc.Doc>` s, all of the
    same language and sharing the same ``spacy.Language`` models and vocabulary.
    Track corpus statistics; flexibly add, iterate through, filter for, and
    remove documents; save and load parsed content and metadata to and from
    disk; and more.

    Initialize from a stream of texts and corresponding metadatas::

        >>> cw = textacy.datasets.CapitolWords()
        >>> records = cw.docs(limit=50)
        >>> text_stream, metadata_stream = textacy.io.split_records(
        ...     records, 'text')
        >>> corpus = textacy.Corpus(
        ...     'en', texts=text_stream, metadatas=metadata_stream)
        >>> print(corpus)
        Corpus(50 docs; 32163 tokens)

    Index, slice, and flexibly get particular documents::

        >>> corpus[0]
        Doc(159 tokens; "Mr. Speaker, 480,000 Federal employees are work...")
        >>> corpus[:3]
        [Doc(159 tokens; "Mr. Speaker, 480,000 Federal employees are work..."),
         Doc(219 tokens; "Mr. Speaker, a relationship, to work and surviv..."),
         Doc(336 tokens; "Mr. Speaker, I thank the gentleman for yielding...")]
        >>> match_func = lambda doc: doc.metadata['speaker_name'] == 'Bernie Sanders'
        >>> for doc in corpus.get(match_func, limit=3):
        ...     print(doc)
        Doc(159 tokens; "Mr. Speaker, 480,000 Federal employees are work...")
        Doc(336 tokens; "Mr. Speaker, I thank the gentleman for yielding...")
        Doc(177 tokens; "Mr. Speaker, if we want to understand why in th...")

    Add and remove documents, with automatic updating of corpus statistics::

        >>> records = cw.docs(congress=114, limit=25)
        >>> text_stream, metadata_stream = textacy.io.split_records(
        ...     records, 'text')
        >>> corpus.add_texts(text_stream, metadatas=metadata_stream)
        >>> print(corpus)
        Corpus(75 docs; 55869 tokens)
        >>> corpus.remove(lambda doc: doc.metadata['speaker_name'] == 'Rick Santorum')
        >>> print(corpus)
        Corpus(60 docs; 48532 tokens)
        >>> del corpus[:5]
        >>> print(corpus)
        Corpus(55 docs; 47444 tokens)

    Get word and doc frequencies in absolute, relative, or binary form:

        >>> counts = corpus.word_freqs(lemmatize=True, weighting='count')
        >>> idf = corpus.word_doc_freqs(lemmatize=True, weighting='idf')

    Save to and load from disk::

        >>> corpus.save('~/Desktop/congress.pkl')
        >>> corpus = textacy.Corpus.load('~/Desktop/congress.pkl')
        >>> print(corpus)
        Corpus(55 docs; 47444 tokens)

    Args:
        lang (str or ``spacy.Language``): Language of content for all docs in
            corpus. Pass a standard 2-letter language code (e.g. "en") or the name
            of a spacy model for the desired language (e.g. "en_core_web_md")
            or an already-instantiated ``spacy.Language`` object. If a str, the
            value is used to instantiate the corresponding ``spacy.Language``
            with all models loaded by default, and the appropriate 2-letter lang
            code is assigned to :attr:`Corpus.lang`.

            **Note:** The ``spacy.Language`` object parses all documents contents
            and sets the :attr:`spacy_vocab` and :attr:`spacy_stringstore`
            attributes. See https://spacy.io/docs/usage/models#available for
            available spacy models.
        texts (Iterable[str]): Stream of documents as (unicode) text, to be
            processed by spaCy and added to the corpus as
            :class:`Doc <textacy.doc.Doc>` s.
        docs (Iterable[:class:`Doc <textacy.doc.Doc>`] or Iterable[``spacy.Doc``]):
            Stream of documents already-processed by spaCy alone or via textacy.
        metadatas (Iterable[dict]): Stream of dictionaries of relevant doc
            metadata. **Note:** This stream must align exactly with ``texts`` or
            ``docs``, or else metadata will be mis-assigned. More concretely,
            the first item in ``metadatas`` will be assigned to the first item
            in ``texts`` or ``docs``, and so on from there.

    Attributes:
        lang (str): 2-letter code for language of documents in :class:`Corpus`.
        n_docs (int): Number of documents in :class:`Corpus`.
        n_tokens (int): Total number of tokens of all documents in :class:`Corpus`.
        n_sents (int): Total number of sentences of all documents in :class:`Corpus`.
            If the ``spacy.Language`` used to process documents did not include
            a syntactic parser, upon which sentence segmentation relies, this
            value will be null.
        docs (List[:class:`Doc <textacy.doc.Doc>`]): List of documents in
            :class:`Corpus`. In 99% of cases, you should never have to interact
            directly with this list; instead, index and slice directly on
            :class:`Corpus` or use the flexible :meth:`Corpus.get() <Corpus.get>`
            and :meth:`Corpus.remove() <Corpus.remove>` methods.
        spacy_lang (``spacy.Language``): http://spacy.io/docs/#english
        spacy_vocab (``spacy.Vocab``): https://spacy.io/docs#vocab
        spacy_stringstore (``spacy.StringStore``): https://spacy.io/docs#stringstore
    """

    def __init__(self, lang, texts=None, docs=None, metadatas=None):
        if isinstance(lang, compat.unicode_):
            self.spacy_lang = cache.load_spacy(lang)
        elif isinstance(lang, SpacyLang):
            self.spacy_lang = lang
        else:
            raise TypeError(
                "`lang` must be {}, not {}".format(
                    {compat.unicode_, SpacyLang}, type(lang)
                )
            )
        self.lang = self.spacy_lang.lang
        self.spacy_vocab = self.spacy_lang.vocab
        self.spacy_stringstore = self.spacy_vocab.strings
        self.docs = []
        self.n_docs = 0
        self.n_tokens = 0
        # sentence segmentation requires parse; if not available, skip it
        if self.spacy_lang.has_pipe("parser") or any(
            isinstance(pipe[1], DependencyParser) for pipe in self.spacy_lang.pipeline
        ):
            self.n_sents = 0
        else:
            self.n_sents = None

        if texts and docs:
            msg = (
                "Corpus may be initialized with either `texts` or `docs`, but not both."
            )
            raise ValueError(msg)
        if texts:
            self.add_texts(texts, metadatas=metadatas)
        elif docs:
            if metadatas:
                for doc, metadata in compat.zip_(docs, metadatas):
                    self.add_doc(doc, metadata=metadata)
            else:
                for doc in docs:
                    self.add_doc(doc)

    def __repr__(self):
        return "Corpus({} docs; {} tokens)".format(self.n_docs, self.n_tokens)

    def __len__(self):
        return self.n_docs

    def __iter__(self):
        for doc in self.docs:
            yield doc

    def __getitem__(self, idx_or_slice):
        return self.docs[idx_or_slice]

    def __delitem__(self, idx_or_slice):
        if isinstance(idx_or_slice, int):
            self._remove_one_doc_by_index(idx_or_slice)
        elif isinstance(idx_or_slice, slice):
            start, end, step = idx_or_slice.indices(self.n_docs)
            indexes = compat.range_(start, end, step)
            self._remove_many_docs_by_index(indexes)
        else:
            msg = 'value must be {}, not "{}"'.format({int, slice}, type(idx_or_slice))
            raise ValueError(msg)

    @property
    def vectors(self):
        """Constituent docs' word vectors stacked together in a matrix."""
        return np.vstack((doc.spacy_doc.vector for doc in self))

    ##########
    # FILEIO #

    def save(self, filepath):
        """
        Save :class:`Corpus` documents' content and metadata to disk,
        as a ``pickle`` file.

        Args:
            filepath (str): Full path to file on disk where documents' content and
                metadata are to be saved.

        See Also:
            :meth:`Corpus.load()`
        """
        # HACK: add spacy language metadata to first doc's user_data
        # so we can re-instantiate the same language upon Corpus.load()
        self[0].spacy_doc.user_data["textacy"]["spacy_lang_meta"] = self.spacy_lang.meta
        io.write_spacy_docs((doc.spacy_doc for doc in self), filepath)

    @classmethod
    def load(cls, filepath):
        """
        Load documents' pickled content and metadata from disk, and initialize
        a :class:`Corpus` with a spacy language pipeline equivalent to what was
        in use previously, when the corpus was saved.

        Args:
            filepath (str): Full path to file on disk where documents' content and
                metadata are saved.

        Returns:
            :class:`Corpus`

        See Also:
            :meth:`Corpus.save()`
        """
        spacy_docs = io.read_spacy_docs(filepath)
        # HACK: pop spacy language metadata from first doc's user_data
        # so we can (more or less...) re-instantiate the same language pipeline
        first_spacy_doc, spacy_docs = itertoolz.peek(spacy_docs)
        spacy_lang_meta = first_spacy_doc.user_data["textacy"].pop("spacy_lang_meta")
        # manually instantiate the spacy language pipeline and
        # hope that the spacy folks either make this easier or don't touch it
        spacy_lang = get_lang_class(spacy_lang_meta["lang"])(
            vocab=first_spacy_doc.vocab, meta=spacy_lang_meta
        )
        for name in spacy_lang_meta["pipeline"]:
            spacy_lang.add_pipe(spacy_lang.create_pipe(name))
        return cls(spacy_lang, docs=spacy_docs)

    #################
    # ADD DOCUMENTS #

    def _add_textacy_doc(self, doc):
        doc.corpus_index = self.n_docs
        doc.corpus = self
        self.docs.append(doc)
        self.n_docs += 1
        self.n_tokens += doc.n_tokens
        # sentence segmentation requires parse; if not available, skip it
        if self.spacy_lang.has_pipe("parser") or any(
            isinstance(pipe[1], DependencyParser) for pipe in self.spacy_lang.pipeline
        ):
            self.n_sents += doc.n_sents
        LOGGER.debug("added %s to Corpus[%s]", doc, doc.corpus_index)

    def add_texts(self, texts, metadatas=None, n_threads=None, batch_size=1000):
        """
        Process a stream of texts (and a corresponding stream of metadata dicts,
        optionally) in parallel with spaCy; add as :class:`Doc <textacy.doc.Doc>` s
        to the corpus.

        Args:
            texts (Iterable[str]): Stream of texts to add to corpus as
                :class:`Doc <textacy.doc.Doc>` s.
            metadatas (Iterable[dict]): Stream of dictionaries of relevant
                document metadata.

                .. note:: This stream must align exactly with ``texts``, or
                   metadata will be mis-assigned to texts. More concretely,
                   the first item in ``metadatas`` will be assigned to
                   the first item in ``texts``, and so on from there.

            n_threads (int): Number of threads to use when processing ``texts``
            in parallel, if available. DEPRECATED! This never worked correctly
                in spacy v2.0, and no longer does in anything in spacy v2.1.
            batch_size (int): Number of texts to process at a time.

        See Also:
            - :func:`io.split_records() <textacy.io.utils.split_records>`
            - https://spacy.io/api/language#pipe
        """
        if n_threads is not None:
            utils.deprecated(
                "The `n_threads` arg never worked correctly in spacy v2.0, and "
                "doesn't do anything in spacy v2.1, so textacy won't pass it on.",
                action="once",
            )
        spacy_docs = self.spacy_lang.pipe(texts, batch_size=batch_size)
        if metadatas:
            for i, (spacy_doc, metadata) in enumerate(
                compat.zip_(spacy_docs, metadatas)
            ):
                self._add_textacy_doc(
                    Doc(spacy_doc, lang=self.spacy_lang, metadata=metadata)
                )
                if i % batch_size == 0:
                    LOGGER.info("adding texts to %s...", self)
        else:
            for i, spacy_doc in enumerate(spacy_docs):
                self._add_textacy_doc(
                    Doc(spacy_doc, lang=self.spacy_lang, metadata=None)
                )
                if i % batch_size == 0:
                    LOGGER.info("adding texts to %s...", self)

    def add_text(self, text, metadata=None):
        """
        Create a :class:`Doc <textacy.doc.Doc>` from ``text`` and ``metadata``,
        then add it to the corpus.

        Args:
            text (str): Document (text) content to add to corpus as a
                :class:`Doc <textacy.doc.Doc>`.
            metadata (dict): Dictionary of relevant document metadata.
        """
        self._add_textacy_doc(Doc(text, lang=self.spacy_lang, metadata=metadata))

    def add_doc(self, doc, metadata=None):
        """
        Add an existing :class:`Doc <textacy.doc.Doc>` or initialize a
        new one from a ``spacy.Doc`` to the corpus.

        Args:
            doc (:class:`Doc <textacy.doc.Doc>` or ``spacy.Doc``)
            metadata (dict): Dictionary of relevant document metadata. Note:
                If specified, this will *overwrite* any existing metadata.

        Warning:
            If ``doc`` was already added to this or another :class:`Corpus`,
            it will be deep-copied and then added as if a new document. A warning
            message will be logged. This is probably not a thing you should do.
        """
        if isinstance(doc, Doc):
            if doc.spacy_vocab is not self.spacy_vocab:
                msg = "Doc.spacy_vocab {} != Corpus.spacy_vocab {}".format(
                    doc.spacy_vocab, self.spacy_vocab
                )
                raise ValueError(msg)
            if hasattr(doc, "corpus_index"):
                doc = copy.deepcopy(doc)
                LOGGER.warning("Doc already associated with a Corpus; adding anyway...")
            if metadata is not None:
                doc.metadata = metadata
            self._add_textacy_doc(doc)
        elif isinstance(doc, SpacyDoc):
            if doc.vocab is not self.spacy_vocab:
                msg = "SpacyDoc.vocab {} != Corpus.spacy_vocab {}".format(
                    doc.vocab, self.spacy_vocab
                )
                raise ValueError(msg)
            metadata = metadata or doc.user_data.get("textacy", {}).get("metadata")
            self._add_textacy_doc(Doc(doc, lang=self.spacy_lang, metadata=metadata))
        else:
            msg = '`doc` must be {}, not "{}"'.format({Doc, SpacyDoc}, type(doc))
            raise ValueError(msg)

    #################
    # GET DOCUMENTS #

    def get(self, match_func, limit=-1):
        """
        Iterate over docs in :class:`Corpus` and return all (or N <= ``limit``)
        for which ``match_func(doc)`` is True.

        Args:
            match_func (func): Function that takes a :class:`Doc <textacy.doc.Doc>`
                as input and returns a boolean value. For example::

                    Corpus.get(lambda x: len(x) >= 100)

                gets all docs with 100+ tokens. And::

                    Corpus.get(lambda x: x.metadata['author'] == 'Burton DeWilde')

                gets all docs whose author was given as 'Burton DeWilde'.
            limit (int): Maximum number of matched docs to return.

        Yields:
            :class:`Doc <textacy.doc.Doc>`: Next document passing
            ``match_func`` up to ``limit`` docs.

        See Also:
            :meth:`Corpus.remove()`

        .. tip:: To get doc(s) by index, treat :class:`Corpus` as a list and use
           Python's usual indexing and slicing: ``Corpus[0]`` gets the first
           document in the corpus; ``Corpus[:5]`` gets the first 5; etc.
        """
        n_matched_docs = 0
        for doc in self:
            if match_func(doc) is True:
                yield doc
                n_matched_docs += 1
                if n_matched_docs == limit:
                    break

    ###############
    # REMOVE DOCS #

    def _remove_one_doc_by_index(self, index):
        doc = self[index]
        n_tokens_removed = doc.n_tokens
        n_sents_removed = doc.n_sents if self.spacy_lang.parser else None
        # actually remove the doc
        del self.docs[index]
        # shift `corpus_index` attribute on docs higher up in the list
        for doc in self[index:]:
            doc.corpus_index -= 1
        # decrement the corpus doc/token/sent counts
        self.n_docs -= 1
        self.n_tokens -= n_tokens_removed
        if n_sents_removed:
            self.n_sents -= n_sents_removed

    def _remove_many_docs_by_index(self, indexes):
        indexes = sorted(indexes, reverse=True)
        n_docs_removed = len(indexes)
        n_sents_removed = 0
        n_tokens_removed = 0
        for index in indexes:
            n_tokens_removed += self[index].n_tokens
            if self.spacy_lang.parser:
                n_sents_removed += self[index].n_sents
            # actually remove the doc
            del self.docs[index]
        # shift the `corpus_index` attribute for all docs at once
        for i, doc in enumerate(self):
            doc.corpus_index = i
        # decrement the corpus doc/sent/token counts
        self.n_docs -= n_docs_removed
        self.n_tokens -= n_tokens_removed
        self.n_sents -= n_sents_removed

    def remove(self, match_func, limit=-1):
        """
        Remove all (or N <= ``limit``) docs in :class:`Corpus` for which
        ``match_func(doc)`` is True. Corpus doc/sent/token counts are adjusted
        accordingly, as are the :attr:`Doc.corpus_index <textacy.doc.Doc.corpus_index>`
        attributes on affected documents.

        Args:
            match_func (func): Function that takes a :class:`Doc <textacy.doc.Doc>`
                and returns a boolean value. For example::

                    Corpus.remove(lambda x: len(x) >= 100)

                removes docs with 100+ tokens. And::

                    Corpus.remove(lambda x: x.metadata['author'] == 'Burton DeWilde')

                removes docs whose author was given as 'Burton DeWilde'.
            limit (int): Maximum number of matched docs to remove.

        See Also:
            :meth:`Corpus.get()`

        .. tip:: To remove doc(s) by index, treat :class:`Corpus` as a list and use
           Python's usual indexing and slicing: ``del Corpus[0]`` removes the
           first document in the corpus; ``del Corpus[:5]`` removes the first
           5; etc.
        """
        n_matched_docs = 0
        matched_indexes = []
        for i, doc in enumerate(self):
            if match_func(doc) is True:
                matched_indexes.append(i)
                n_matched_docs += 1
                if n_matched_docs == limit:
                    break
        self._remove_many_docs_by_index(matched_indexes)

    def word_freqs(self, normalize="lemma", weighting="count", as_strings=False):
        """
        Map the set of unique words in :class:`Corpus` to their counts as absolute,
        relative, or binary frequencies of occurence. This is akin to
        :func:`Doc.to_bag_of_words() <textacy.doc.Doc.to_bag_of_words>`.

        Args:
            normalize (str): if 'lemma', lemmatize words before counting; if
                'lower', lowercase words before counting; otherwise, words are
                counted using the form with which they they appear in docs
            weighting ({'count', 'freq', 'binary'}): Type of weight to assign to
                words. If 'count' (default), weights are the absolute number of
                occurrences (count) of word in corpus. If 'binary', all counts
                are set equal to 1. If 'freq', word counts are normalized by the
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

        See Also:
            :func:`vsm.get_term_freqs() <textacy.vsm.get_term_freqs>``
        """
        word_counts = collections.Counter()
        for doc in self:
            word_counts.update(
                doc.to_bag_of_words(
                    normalize=normalize, weighting="count", as_strings=as_strings
                )
            )
        if weighting == "count":
            word_counts = dict(word_counts)
        if weighting == "freq":
            n_tokens = self.n_tokens
            word_counts = {
                word: weight / n_tokens for word, weight in word_counts.items()
            }
        elif weighting == "binary":
            word_counts = {word: 1 for word in word_counts.keys()}
        return word_counts

    def word_doc_freqs(
        self, normalize="lemma", weighting="count", smooth_idf=True, as_strings=False
    ):
        """
        Map the set of unique words in :class:`Corpus` to their *document* counts
        as absolute, relative, inverse, or binary frequencies of occurence.

        Args:
            normalize (str): if 'lemma', lemmatize words before counting; if
                'lower', lowercase words before counting; otherwise, words are
                counted using the form with which they they appear in docs
            weighting ({'count', 'freq', 'idf', 'binary'}): Type of weight to
                assign to words. If 'count' (default), weights are the absolute
                number (count) of documents in which word appears. If 'binary',
                all counts are set equal to 1. If 'freq', word doc counts are
                normalized by the total document count, giving their relative
                frequency of occurrence. If 'idf', weights are the log of the
                inverse relative frequencies: ``log(n_docs / word_doc_count)``
                or ``log(1 + n_docs / word_doc_count)`` if ``smooth_idf`` is True.
            smooth_idf (bool): if True, add 1 to all document frequencies when
                calculating 'idf' weighting, equivalent to adding a single
                document to the corpus containing every unique word
            as_strings (bool): if True, words are returned as strings; if False
                (default), words are returned as their unique integer ids

        Returns:
            dict: mapping of a unique word id or string (depending on the value
            of ``as_strings``) to the number of documents in which it appears
            weighted as absolute, relative, or binary frequencies (depending
            on the value of ``weighting``).

        See Also:
            :func:`vsm.get_doc_freqs() <textacy.vsm.matrix_utils.get_doc_freqs>`
        """
        word_doc_counts = collections.Counter()
        for doc in self:
            word_doc_counts.update(
                doc.to_bag_of_words(
                    normalize=normalize, weighting="binary", as_strings=as_strings
                )
            )
        if weighting == "count":
            word_doc_counts = dict(word_doc_counts)
        elif weighting == "freq":
            n_docs = self.n_docs
            word_doc_counts = {
                word: count / n_docs for word, count in word_doc_counts.items()
            }
        elif weighting == "idf":
            n_docs = self.n_docs
            if smooth_idf is True:
                word_doc_counts = {
                    word: math.log(1 + n_docs / count)
                    for word, count in word_doc_counts.items()
                }
            else:
                word_doc_counts = {
                    word: math.log(n_docs / count)
                    for word, count in word_doc_counts.items()
                }
        elif weighting == "binary":
            word_doc_counts = {word: 1 for word in word_doc_counts.keys()}
        return word_doc_counts
