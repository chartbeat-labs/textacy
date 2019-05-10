# -*- coding: utf-8 -*-
"""
Corpus
------

An ordered collection of :class:`spacy.tokens.Doc`, all of the same language
and sharing the same :class:`spacy.language.Language` processing pipeline
and vocabulary, with data held *in-memory*. Includes functionality for
easily adding, getting, and removing of documents; saving to / loading from disk;
and tracking basic corpus statistics.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import io
import itertools
import logging

import numpy as np
import spacy
import srsly
from cytoolz import itertoolz
from thinc.neural.ops import NumpyOps

from . import cache
from . import compat
from . import io as tio
from . import utils


class Corpus(object):
    """
    An ordered collection of :class:`spacy.tokens.Doc`, all of the same language
    and sharing the same :class:`spacy.language.Language` processing pipeline
    and vocabulary, with data held *in-memory*.

    Initialize from a stream of texts or (text, metadata) pairs::

        >>> ds = textacy.datasets.CapitolWords()
        >>> records = ds.records(limit=50)
        >>> corpus = textacy.Corpus("en", data=records)
        >>> print(corpus)
        Corpus(50 docs; 32163 tokens)

    Index, slice, and flexibly get particular documents::

        >>> corpus[0]
        Doc(159 tokens; "Mr. Speaker, 480,000 Federal employees are work...")
        >>> corpus[:3]
        [Doc(159 tokens; "Mr. Speaker, 480,000 Federal employees are work..."),
         Doc(219 tokens; "Mr. Speaker, a relationship, to work and surviv..."),
         Doc(336 tokens; "Mr. Speaker, I thank the gentleman for yielding...")]
        >>> match_func = lambda doc: doc._.meta["speaker_name"] == "Bernie Sanders"
        >>> for doc in corpus.get(match_func, limit=3):
        ...     print(doc)
        Doc(159 tokens; "Mr. Speaker, 480,000 Federal employees are work...")
        Doc(336 tokens; "Mr. Speaker, I thank the gentleman for yielding...")
        Doc(177 tokens; "Mr. Speaker, if we want to understand why in th...")

    Add and remove documents, with automatic updating of corpus statistics::

        >>> records = ds.records(congress=114, limit=25)
        >>> corpus.add(records)
        >>> print(corpus)
        Corpus(75 docs; 55869 tokens)
        >>> corpus.remove(lambda doc: doc._.meta["speaker_name"] == "Rick Santorum")
        >>> print(corpus)
        Corpus(60 docs; 48532 tokens)
        >>> del corpus[:5]
        >>> print(corpus)
        Corpus(55 docs; 47444 tokens)

    Get word and doc frequencies in absolute, relative, or binary form? TBD.

    Save corpus data to and load from disk::

        >>> corpus.save("~/Desktop/capitol_words_sample.bin")
        >>> corpus = textacy.Corpus.load("en", "~/Desktop/capitol_words_sample.bin")
        >>> print(corpus)
        Corpus(55 docs; 47444 tokens)

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
            self._remove_doc_by_index(idx_or_slice)
        elif isinstance(idx_or_slice, slice):
            start, end, step = idx_or_slice.indices(self.n_docs)
            for idx in compat.range_(start, end, step):
                self._remove_doc_by_index(idx)
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
        for doc in itertools.islice(matched_docs, limit):
            self._remove_doc_by_index(self._doc_ids.index(id(doc)))

    def _remove_doc_by_index(self, idx):
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

    # word_freqs() ?
    # word_doc_freqs() ?

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
        # TODO: handle document metadata!
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
        for doc in self:
            tokens.append(doc.to_array(attrs))
            lengths.append(len(doc))
            strings.update(tok.text for tok in doc)

        msg = {
            "meta": self.spacy_lang.meta,
            "attrs": attrs,
            "tokens": np.vstack(tokens).tobytes("C"),
            "lengths": np.asarray(lengths, dtype="int32").tobytes("C"),
            "strings": list(strings),
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
            logging.warning("the spacy langs are different!")
        for string in msg["strings"]:
            spacy_lang.vocab[string]
        attrs = msg["attrs"]
        lengths = np.frombuffer(msg["lengths"], dtype="int32")
        flat_tokens = np.frombuffer(msg["tokens"], dtype="uint64")
        flat_tokens = flat_tokens.reshape(
            (flat_tokens.size // len(attrs), len(attrs))
        )
        tokens = np.asarray(NumpyOps().unflatten(flat_tokens, lengths))

        def _make_docs(tokens):
            for toks in tokens:
                doc = spacy.tokens.Doc(
                    spacy_lang.vocab,
                    words=[spacy_lang.vocab.strings[orth] for orth in toks[:, 0]],
                    spaces=np.ndarray.tolist(toks[:, 1]),
                )
                doc = doc.from_array(attrs[2:], toks[:, 2:])
                yield doc

        return cls(spacy_lang, data=_make_docs(tokens))


def _get_spacy_lang(lang):
    if isinstance(lang, compat.unicode_):
        return cache.load_spacy(lang)
    elif isinstance(lang, spacy.language.Language):
        return lang
    else:
        raise TypeError(
            "`lang` must be {}, not {}".format(
                {compat.unicode_, spacy.language.Language}, type(lang)
            )
        )
