# -*- coding: utf-8 -*-
"""
A class for working with a collection of spaCy docs. Includes functionality for
easily adding, getting, and removing documents; saving to / loading their data
from disk; and tracking basic corpus statistics.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import itertools
import logging
import math

import numpy as np
import spacy
import srsly
from cytoolz import itertoolz
from thinc.neural.ops import NumpyOps

from . import cache
from . import compat
from . import io as tio
from . import utils

LOGGER = logging.getLogger(__name__)


class Corpus(object):
    """
    An ordered collection of :class:`spacy.tokens.Doc`, all of the same language
    and sharing the same :class:`spacy.language.Language` processing pipeline
    and vocabulary, with data held *in-memory*.

    Initialize from a language / ``Language`` and (optionally) one or a stream
    of texts or (text, metadata) pairs:

    .. code-block:: pycon

        >>> ds = textacy.datasets.CapitolWords()
        >>> records = ds.records(limit=50)
        >>> corpus = textacy.Corpus("en", data=records)
        >>> corpus
        Corpus(50 docs, 32175 tokens)

    Add or remove documents, with automatic updating of corpus statistics:

    .. code-block:: pycon

        >>> texts = ds.texts(congress=114, limit=25)
        >>> corpus.add(texts)
        >>> corpus.add("If Burton were a member of Congress, here's what he'd say.")
        >>> corpus
        Corpus(76 docs, 55906 tokens)
        >>> corpus.remove(lambda doc: doc._.meta.get("speaker_name") == "Rick Santorum")
        >>> corpus
        Corpus(61 docs, 48567 tokens)

    Get subsets of documents matching your particular use case:

    .. code-block:: pycon

        >>> match_func = lambda doc: doc._.meta.get("speaker_name") == "Bernie Sanders"
        >>> for doc in corpus.get(match_func, limit=3):
        ...     print(doc._.preview)
        Doc(159 tokens: "Mr. Speaker, 480,000 Federal employees are work...")
        Doc(336 tokens: "Mr. Speaker, I thank the gentleman for yielding...")
        Doc(177 tokens: "Mr. Speaker, if we want to understand why in th...")

    Get or remove documents by indexing, too:

    .. code-block:: pycon

        >>> corpus[0]._.preview
        'Doc(159 tokens: "Mr. Speaker, 480,000 Federal employees are work...")'
        >>> [doc._.preview for doc in corpus[:3]]
        ['Doc(159 tokens: "Mr. Speaker, 480,000 Federal employees are work...")',
         'Doc(219 tokens: "Mr. Speaker, a relationship, to work and surviv...")',
         'Doc(336 tokens: "Mr. Speaker, I thank the gentleman for yielding...")']
        >>> del corpus[:5]
        >>> corpus
        Corpus(56 docs, 41573 tokens)

    Compute basic corpus statistics:

    .. code-block:: pycon

        >>> corpus.n_docs, corpus.n_sents, corpus.n_tokens
        (56, 1771, 41573)
        >>> word_counts = corpus.word_counts(as_strings=True)
        >>> sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        [('-PRON-', 2553), ('people', 215), ('year', 148), ('Mr.', 139), ('$', 137)]
        >>> word_doc_counts = corpus.word_doc_counts(weighting="freq", as_strings=True)
        >>> sorted(word_doc_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        [('-PRON-', 0.9821428571428571),
         ('Mr.', 0.7678571428571429),
         ('President', 0.5),
         ('people', 0.48214285714285715),
         ('need', 0.44642857142857145)]

    Save corpus data to and load from disk:

    .. code-block:: pycon

        >>> corpus.save("~/Desktop/capitol_words_sample.bin.gz")
        >>> corpus = textacy.Corpus.load("en", "~/Desktop/capitol_words_sample.bin.gz")
        >>> corpus
        Corpus(56 docs, 41573 tokens)

    Args:
        lang (str or :class:`spacy.language.Language`):
            Language with which spaCy processes (or processed) all documents
            added to the corpus, whether as ``data`` now or later.

            Pass a standard 2-letter language code (e.g. "en"),
            or the name of a spacy language pipeline (e.g. "en_core_web_md"),
            or an already-instantiated :class:`spacy.language.Language` object.

            A given / detected language string is then used to instantiate
            a corresponding ``Language`` with all default components enabled.
        data (obj or Iterable[obj]): One or a stream of texts, records,
            or :class:`spacy.tokens.Doc` s to be added to the corpus.

            .. seealso:: :meth:`Corpus.add()`

    Attributes:
        lang (str)
        spacy_lang (:class:`spacy.language.Language`)
        docs (List[:class:`spacy.tokens.Doc`])
        n_docs (int)
        n_sents (int)
        n_tokens (int)
    """

    def __init__(self, lang, data=None):
        self.spacy_lang = _get_spacy_lang(lang)
        self.lang = self.spacy_lang.lang
        self.docs = []
        self._doc_ids = []
        self.n_docs = 0
        self.n_sents = 0
        self.n_tokens = 0
        if data:
            self.add(data)

    # dunder

    def __repr__(self):
        return "Corpus({} docs, {} tokens)".format(self.n_docs, self.n_tokens)

    def __len__(self):
        return self.n_docs

    def __iter__(self):
        for doc in self.docs:
            yield doc

    def __contains__(self, doc):
        return id(doc) in self._doc_ids

    def __getitem__(self, idx_or_slice):
        return self.docs[idx_or_slice]

    def __delitem__(self, idx_or_slice):
        if isinstance(idx_or_slice, int):
            self._remove_one_doc_by_index(idx_or_slice)
        elif isinstance(idx_or_slice, slice):
            start, end, step = idx_or_slice.indices(self.n_docs)
            idxs = compat.range_(start, end, step)
            self._remove_many_docs_by_index(idxs)
        else:
            raise TypeError(
                "list indices must be integers or slices, not {}".format(type(idx_or_slice))
            )

    # add documents

    def add(self, data, batch_size=1000):
        """
        Add one or a stream of texts, records, or :class:`spacy.tokens.Doc` s
        to the corpus, ensuring that all processing is or has already been done
        by the :attr:`Corpus.spacy_lang` pipeline.

        Args:
            data (obj or Iterable[obj]):
                str or Iterable[str]
                Tuple[str, dict] or Iterable[Tuple[str, dict]]
                :class:`spacy.tokens.Doc` or Iterable[:class:`spacy.tokens.Doc`]
            batch_size (int)

        See Also:
            * :meth:`Corpus.add_text()`
            * :meth:`Corpus.add_texts()`
            * :meth:`Corpus.add_record()`
            * :meth:`Corpus.add_records()`
            * :meth:`Corpus.add_doc()`
            * :meth:`Corpus.add_docs()`
        """
        if isinstance(data, compat.unicode_):
            self.add_text(data)
        elif isinstance(data, spacy.tokens.Doc):
            self.add_doc(data)
        elif utils.is_record(data):
            self.add_record(data)
        elif isinstance(data, compat.Iterable):
            first, data = itertoolz.peek(data)
            if isinstance(first, compat.unicode_):
                self.add_texts(data, batch_size=batch_size)
            elif isinstance(first, spacy.tokens.Doc):
                self.add_docs(data)
            elif utils.is_record(first):
                self.add_records(data, batch_size=batch_size)
            else:
                raise TypeError(
                    "data must be one of {} or an interable thereof, not {}".format(
                        {compat.unicode_, spacy.tokens.Doc, tuple},
                        type(data),
                    )
                )
        else:
            raise TypeError(
                "data must be one of {} or an interable thereof, not {}".format(
                    {compat.unicode_, spacy.tokens.Doc, tuple},
                    type(data),
                )
            )

    def add_text(self, text):
        """
        Add one text to the corpus, processing it into a :class:`spacy.tokens.Doc`
        using the :attr:`Corpus.spacy_lang` pipeline.

        Args:
            text (str)
        """
        self._add_valid_doc(self.spacy_lang(text))

    def add_texts(self, texts, batch_size=1000):
        """
        Add a stream of texts to the corpus, efficiently processing them into
        :class:`spacy.tokens.Doc` s using the :attr:`Corpus.spacy_lang` pipeline.

        Args:
            texts (Iterable[str])
            batch_size (int)
        """
        for doc in self.spacy_lang.pipe(texts, as_tuples=False, batch_size=batch_size):
            self._add_valid_doc(doc)

    def add_record(self, record):
        """
        Add one record to the corpus, processing it into a :class:`spacy.tokens.Doc`
        using the :attr:`Corpus.spacy_lang` pipeline.

        Args:
            record (Tuple[str, dict])
        """
        doc = self.spacy_lang(record[0])
        doc._.meta = record[1]
        self._add_valid_doc(doc)

    def add_records(self, records, batch_size=1000):
        """
        Add a stream of records to the corpus, efficiently processing them into
        :class:`spacy.tokens.Doc` s using the :attr:`Corpus.spacy_lang` pipeline.

        Args:
            records (Iterable[Tuple[str, dict]])
            batch_size (int)
        """
        for doc, meta in self.spacy_lang.pipe(records, as_tuples=True, batch_size=batch_size):
            doc._.meta = meta
            self._add_valid_doc(doc)

    def add_doc(self, doc):
        """
        Add one :class:`spacy.tokens.Doc` to the corpus, provided it was processed
        using the :attr:`Corpus.spacy_lang` pipeline.

        Args:
            doc (:class:`spacy.tokens.Doc`)
        """
        if not isinstance(doc, spacy.tokens.Doc):
            raise TypeError(
                "doc must be a {}, not {}".format(spacy.tokens.Doc, type(doc))
            )
        if doc.vocab is not self.spacy_lang.vocab:
            raise ValueError(
                "doc.vocab ({}) must be the same as corpus.vocab ({})".format(
                    doc.vocab, self.spacy_lang.vocab,
                )
            )
        self._add_valid_doc(doc)

    def add_docs(self, docs):
        """
        Add a stream of :class:`spacy.tokens.Doc` s to the corpus, provided
        they were processed using the :attr:`Corpus.spacy_lang` pipeline.

        Args:
            doc (Iterable[:class:`spacy.tokens.Doc`])
        """
        for doc in docs:
            self.add_doc(doc)

    def _add_valid_doc(self, doc):
        self.docs.append(doc)
        self._doc_ids.append(id(doc))
        self.n_docs += 1
        self.n_tokens += len(doc)
        if doc.is_sentenced:
            self.n_sents += sum(1 for _ in doc.sents)

    # get documents

    def get(self, match_func, limit=None):
        """
        Get all (or N <= ``limit``) docs in :class:`Corpus` for which
        ``match_func(doc)`` is True.

        Args:
            match_func (Callable): Function that takes a :class:`spacy.tokens.Doc`
                as input and returns a boolean value. For example::

                    Corpus.get(lambda x: len(x) >= 100)

                gets all docs with at least 100 tokens. And::

                    Corpus.get(lambda doc: doc._.meta["author"] == "Burton DeWilde")

                gets all docs whose author was given as 'Burton DeWilde'.
            limit (int): Maximum number of matched docs to return.

        Yields:
            :class:`spacy.tokens.Doc`: Next document passing ``match_func``.

        See Also:
            :meth:`Corpus.remove()`

        .. tip:: To get doc(s) by index, treat :class:`Corpus` as a list and use
           Python's usual indexing and slicing: ``Corpus[0]`` gets the first
           document in the corpus; ``Corpus[:5]`` gets the first 5; etc.
        """
        matched_docs = (doc for doc in self if match_func(doc) is True)
        for doc in itertools.islice(matched_docs, limit):
            yield doc

    # remove documents

    def remove(self, match_func, limit=None):
        """
        Remove all (or N <= ``limit``) docs in :class:`Corpus` for which
        ``match_func(doc)`` is True. Corpus doc/sent/token counts are adjusted
        accordingly.

        Args:
            match_func (func): Function that takes a :class:`spacy.tokens.Doc`
                and returns a boolean value. For example::

                    Corpus.remove(lambda x: len(x) >= 100)

                removes docs with at least 100 tokens. And::

                    Corpus.remove(lambda doc: doc._.meta["author"] == "Burton DeWilde")

                removes docs whose author was given as "Burton DeWilde".
            limit (int): Maximum number of matched docs to remove.

        See Also:
            :meth:`Corpus.get()`

        .. tip:: To remove doc(s) by index, treat :class:`Corpus` as a list and use
           Python's usual indexing and slicing: ``del Corpus[0]`` removes the
           first document in the corpus; ``del Corpus[:5]`` removes the first
           5; etc.
        """
        matched_docs = (doc for doc in self if match_func(doc) is True)
        self._remove_many_docs_by_index(
            self._doc_ids.index(id(doc))
            for doc in itertools.islice(matched_docs, limit)
        )

    def _remove_many_docs_by_index(self, idxs):
        for idx in sorted(idxs, reverse=True):
            self._remove_one_doc_by_index(idx)

    def _remove_one_doc_by_index(self, idx):
        doc = self.docs[idx]
        self.n_docs -= 1
        self.n_tokens -= len(doc)
        if doc.is_sentenced:
            self.n_sents -= sum(1 for _ in doc.sents)
        del self.docs[idx]
        del self._doc_ids[idx]

    # useful properties

    @property
    def vectors(self):
        """Constituent docs' word vectors stacked in a 2d array."""
        return np.vstack((doc.vector for doc in self))

    @property
    def vector_norms(self):
        """Constituent docs' L2-normalized word vectors stacked in a 2d array."""
        return np.vstack((doc.vector_norm for doc in self))

    # useful methods

    def word_counts(self, normalize="lemma", weighting="count", as_strings=False):
        """
        Map the set of unique words in :class:`Corpus` to their counts as
        absolute, relative, or binary frequencies of occurence,
        similar to :meth:`Doc._.to_bag_of_words()` but aggregated over all docs.

        Args:
            normalize (str): If "lemma", lemmatize words before counting; if
                "lower", lowercase words before counting; otherwise, words are
                counted using the form with which they appear.
            weighting ({"count", "freq"}): Type of weight to assign to words.
                If "count" (default), weights are the absolute number of
                occurrences (count) of word in corpus.
                If "freq", word counts are normalized by the total token count,
                giving their relative frequencies of occurrence.

                .. note:: The resulting set of frequencies won't (necessarily) sum
                   to 1.0, since punctuation and stop words are filtered out after
                   counts are normalized.

            as_strings (bool): If True, words are returned as strings; if False
                (default), words are returned as their unique integer ids.

        Returns:
            dict: mapping of a unique word id or string (depending on the value
            of ``as_strings``) to its absolute, relative, or binary frequency
            of occurrence (depending on the value of ``weighting``).

        See Also:
            :func:`textacy.vsm.get_term_freqs()`
        """
        word_counts_ = collections.Counter()
        for doc in self:
            word_counts_.update(
                doc._.to_bag_of_words(
                    normalize=normalize, weighting="count", as_strings=as_strings
                )
            )
        if weighting == "count":
            word_counts_ = dict(word_counts_)
        elif weighting == "freq":
            n_tokens = self.n_tokens
            word_counts_ = {
                word: count / n_tokens for word, count in word_counts_.items()
            }
        else:
            raise ValueError(
                "weighting='{}' is invalid; valid values are {}".format(
                    weighting, {"count", "freq"}
                )
            )
        return word_counts_

    def word_doc_counts(
        self, normalize="lemma", weighting="count", smooth_idf=True, as_strings=False
    ):
        """
        Map the set of unique words in :class:`Corpus` to their *document* counts
        as absolute, relative, inverse, or binary frequencies of occurence.

        Args:
            normalize (str): If "lemma", lemmatize words before counting; if
                "lower", lowercase words before counting; otherwise, words are
                counted using the form with which they appear.
            weighting ({"count", "freq", "idf"}): Type of weight to assign to words.
                If "count" (default), weights are the absolute number (count)
                of documents in which word appears. If "freq", word doc counts
                are normalized by the total document count, giving their relative
                frequencies of occurrence. If "idf", weights are the log of the
                inverse relative frequencies: ``log(n_docs / word_doc_count)``
                or (if ``smooth_idf`` is True) ``log(1 + (n_docs / word_doc_count))`` .
            smooth_idf (bool): If True, add 1 to all word doc counts when
                calculating "idf" weighting, equivalent to adding a single
                document to the corpus containing every unique word.
            as_strings (bool): If True, words are returned as strings; if False
                (default), words are returned as their unique integer ids

        Returns:
            dict: mapping of a unique word id or string (depending on the value
            of ``as_strings``) to the number of documents in which it appears
            weighted as absolute, relative, or binary frequencies (depending
            on the value of ``weighting``).

        See Also:
            :func:`textacy.vsm.get_doc_freqs()`
        """
        word_doc_counts_ = collections.Counter()
        for doc in self:
            word_doc_counts_.update(
                doc._.to_bag_of_words(
                    normalize=normalize, weighting="binary", as_strings=as_strings
                )
            )
        if weighting == "count":
            word_doc_counts_ = dict(word_doc_counts_)
        elif weighting == "freq":
            n_docs = self.n_docs
            word_doc_counts_ = {
                word: count / n_docs for word, count in word_doc_counts_.items()
            }
        elif weighting == "idf":
            n_docs = self.n_docs
            if smooth_idf is True:
                word_doc_counts_ = {
                    word: math.log(1 + (n_docs / count))
                    for word, count in word_doc_counts_.items()
                }
            else:
                word_doc_counts_ = {
                    word: math.log(n_docs / count)
                    for word, count in word_doc_counts_.items()
                }
        else:
            raise ValueError(
                "weighting='{}' is invalid; valid values are {}".format(
                    weighting, {"count", "freq", "idf"}
                )
            )
        return word_doc_counts_

    # file io

    def save(self, filepath):
        """
        Save :class:`Corpus` to disk as binary data.

        Args:
            filepath (str): Full path to file on disk where :class:`Corpus` data
                will be saved as a binary file.

        See Also:
            :meth:`Corpus.load()`
        """
        attrs = [
            spacy.attrs.ORTH,
            spacy.attrs.SPACY,
            spacy.attrs.LEMMA,
            spacy.attrs.ENT_IOB,
            spacy.attrs.ENT_TYPE,
        ]
        if self[0].is_tagged:
            attrs.append(spacy.attrs.TAG)
        if self[0].is_parsed:
            attrs.append(spacy.attrs.HEAD)
            attrs.append(spacy.attrs.DEP)
        else:
            attrs.append(spacy.attrs.SENT_START)

        tokens = []
        lengths = []
        strings = set()
        user_datas = []
        for doc in self:
            tokens.append(doc.to_array(attrs))
            lengths.append(len(doc))
            strings.update(tok.text for tok in doc)
            user_datas.append(doc.user_data)

        msg = {
            "meta": self.spacy_lang.meta,
            "attrs": attrs,
            "tokens": np.vstack(tokens).tobytes("C"),
            "lengths": np.asarray(lengths, dtype="int32").tobytes("C"),
            "strings": list(strings),
            "user_datas": user_datas,
        }
        with tio.open_sesame(filepath, mode="wb") as f:
            f.write(srsly.msgpack_dumps(msg))

    @classmethod
    def load(cls, lang, filepath):
        """
        Load previously saved :class:`Corpus` binary data, reproduce the original
        `:class:`spacy.tokens.Doc`s tokens and annotations, and instantiate
        a new :class:`Corpus` from them.

        Args:
            lang (str or :class:`spacy.language.Language`)
            filepath (str): Full path to file on disk where :class:`Corpus` data
                was previously saved as a binary file.

        Returns:
            :class:`Corpus`

        See Also:
            :meth:`Corpus.save()`
        """
        spacy_lang = _get_spacy_lang(lang)
        with tio.open_sesame(filepath, mode="rb") as f:
            msg = srsly.msgpack_loads(f.read())
        if spacy_lang.meta != msg["meta"]:
            LOGGER.warning("the spacy langs are different!")
        for string in msg["strings"]:
            spacy_lang.vocab[string]
        attrs = msg["attrs"]
        lengths = np.frombuffer(msg["lengths"], dtype="int32")
        flat_tokens = np.frombuffer(msg["tokens"], dtype="uint64")
        flat_tokens = flat_tokens.reshape(
            (flat_tokens.size // len(attrs), len(attrs))
        )
        tokens = np.asarray(NumpyOps().unflatten(flat_tokens, lengths))
        user_datas = msg["user_datas"]

        def _make_spacy_docs(tokens, user_datas):
            for toks, user_data in compat.zip_(tokens, user_datas):
                doc = spacy.tokens.Doc(
                    spacy_lang.vocab,
                    words=[spacy_lang.vocab.strings[orth] for orth in toks[:, 0]],
                    spaces=np.ndarray.tolist(toks[:, 1]),
                )
                doc = doc.from_array(attrs[2:], toks[:, 2:])
                doc.user_data = user_data
                yield doc

        return cls(spacy_lang, data=_make_spacy_docs(tokens, user_datas))


def _get_spacy_lang(lang):
    if isinstance(lang, compat.unicode_):
        return cache.load_spacy_lang(lang)
    elif isinstance(lang, spacy.language.Language):
        return lang
    else:
        raise TypeError(
            "`lang` must be {}, not {}".format(
                {compat.unicode_, spacy.language.Language}, type(lang)
            )
        )
