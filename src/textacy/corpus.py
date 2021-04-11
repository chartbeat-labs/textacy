"""
:mod:`textacy.corpus`: Class for working with a collection of spaCy ``Doc`` s.
Includes functionality for easily adding, getting, and removing documents;
saving to / loading their data from disk; and tracking basic corpus statistics.
"""
from __future__ import annotations

import collections
import itertools
import logging
import math
from typing import (
    Any,
    Callable,
    Counter,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Union,
)

import numpy as np
import spacy
from cytoolz import itertoolz
from spacy.language import Language
from spacy.tokens import Doc, DocBin

from . import io as tio
from . import errors, extensions, spacier, types, utils


LOGGER = logging.getLogger(__name__)


class Corpus:
    """
    An ordered collection of :class:`spacy.tokens.Doc`, all of the same language
    and sharing the same :class:`spacy.language.Language` processing pipeline
    and vocabulary, with data held *in-memory*.

    Initialize from a ``Language`` name or instance and (optionally) one or a stream
    of texts or (text, metadata) pairs:

    .. code-block:: pycon

        >>> ds = textacy.datasets.CapitolWords()
        >>> records = ds.records(limit=50)
        >>> corpus = textacy.Corpus("en_core_web_sm", data=records)
        >>> print(corpus)
        Corpus(50 docs, 32175 tokens)

    Add or remove documents, with automatic updating of corpus statistics:

    .. code-block:: pycon

        >>> texts = ds.texts(congress=114, limit=25)
        >>> corpus.add(texts)
        >>> corpus.add("If Burton were a member of Congress, here's what he'd say.")
        >>> print(corpus)
        Corpus(76 docs, 55906 tokens)
        >>> corpus.remove(lambda doc: doc._.meta.get("speaker_name") == "Rick Santorum")
        >>> print(corpus)
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
        >>> print(corpus)
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

        >>> corpus.save("./capitol_words_sample.bin.gz")
        >>> corpus = textacy.Corpus.load("en_core_web_sm", "./capitol_words_sample.bin.gz")
        >>> print(corpus)
        Corpus(56 docs, 41573 tokens)

    Args:
        lang:
            Language with which spaCy processes (or processed) all documents
            added to the corpus, whether as ``data`` now or later.

            Pass the name of a spacy language pipeline (e.g. "en_core_web_sm"),
            or an already-instantiated :class:`spacy.language.Language` object.

            A given / detected language string is then used to instantiate
            a corresponding ``Language`` with all default components enabled.
        data: One or a stream of texts, records, or :class:`spacy.tokens.Doc` s
            to be added to the corpus.

            .. seealso:: :meth:`Corpus.add()`

    Attributes:
        lang
        spacy_lang
        docs
        n_docs
        n_sents
        n_tokens
    """

    lang: str
    spacy_lang: Language
    docs: List[Doc]
    _doc_ids: List[int]
    n_docs: int
    n_sents: int
    n_tokens: int

    def __init__(self, lang: types.LangLike, data: Optional[types.CorpusData] = None):
        self.spacy_lang = spacier.utils.resolve_langlike(lang)
        self.lang = self.spacy_lang.lang
        self.docs = []
        self._doc_ids = []
        self.n_docs = 0
        self.n_sents = 0
        self.n_tokens = 0
        if data is not None:
            self.add(data)

    # dunder

    def __str__(self):
        return f"Corpus({self.n_docs} docs, {self.n_tokens} tokens)"

    def __len__(self):
        return self.n_docs

    def __iter__(self):
        for doc in self.docs:
            yield doc

    def __contains__(self, doc):
        return id(doc) in self._doc_ids

    def __getitem__(self, idx_or_slice):
        return self.docs[idx_or_slice]

    def __delitem__(self, idx_or_slice: int | slice):
        if isinstance(idx_or_slice, int):
            self._remove_one_doc_by_index(idx_or_slice)
        elif isinstance(idx_or_slice, slice):
            start, end, step = idx_or_slice.indices(self.n_docs)
            idxs = range(start, end, step)
            self._remove_many_docs_by_index(idxs)
        else:
            raise TypeError(
                errors.type_invalid_msg(
                    "idx_or_slice", type(idx_or_slice), Union[int, slice]
                )
            )

    # add documents

    def add(self, data: types.CorpusData, batch_size: int = 1000, n_process: int = 1):
        """
        Add one or a stream of texts, records, or :class:`spacy.tokens.Doc` s
        to the corpus, ensuring that all processing is or has already been done
        by the :attr:`Corpus.spacy_lang` pipeline.

        Args:
            data
            batch_size: Number of texts to buffer when processing with spaCy.
            n_process: Number of parallel processors to run when processing.
                If -1, this is set to ``multiprocessing.cpu_count()``.

                .. note:: This feature is only available in spaCy 2.2.2+, and only applies
                   when ``data`` is a sequence of texts or records.

        See Also:
            * :meth:`Corpus.add_text()`
            * :meth:`Corpus.add_texts()`
            * :meth:`Corpus.add_record()`
            * :meth:`Corpus.add_records()`
            * :meth:`Corpus.add_doc()`
            * :meth:`Corpus.add_docs()`
        """
        if isinstance(data, str):
            self.add_text(data)
        elif isinstance(data, Doc):
            self.add_doc(data)
        elif utils.is_record(data):
            self.add_record(data)
        elif isinstance(data, collections.abc.Iterable):
            first, data = itertoolz.peek(data)
            if isinstance(first, str):
                self.add_texts(data, batch_size=batch_size, n_process=n_process)
            elif isinstance(first, Doc):
                self.add_docs(data)
            elif utils.is_record(first):
                self.add_records(data, batch_size=batch_size, n_process=n_process)
            else:
                raise TypeError(
                    errors.type_invalid_msg("data", type(data), types.CorpusData)
                )
        else:
            raise TypeError(
                errors.type_invalid_msg("data", type(data), types.CorpusData)
            )

    def add_text(self, text: str) -> None:
        """
        Add one text to the corpus, processing it into a :class:`spacy.tokens.Doc`
        using the :attr:`Corpus.spacy_lang` pipeline.

        Args:
            text (str)
        """
        self._add_valid_doc(self.spacy_lang(text))

    def add_texts(
        self, texts: Iterable[str], batch_size: int = 1000, n_process: int = 1,
    ) -> None:
        """
        Add a stream of texts to the corpus, efficiently processing them into
        :class:`spacy.tokens.Doc` s using the :attr:`Corpus.spacy_lang` pipeline.

        Args:
            texts: Sequence of texts to process and add to corpus.
            batch_size: Number of texts to buffer when processing with spaCy.
            n_process: Number of parallel processors to run when processing.
                If -1, this is set to ``multiprocessing.cpu_count()``.

                .. note:: This feature is only available in spaCy 2.2.2+.
        """
        if spacy.__version__ >= "2.2.2":
            for doc in self.spacy_lang.pipe(
                texts, as_tuples=False, batch_size=batch_size, n_process=n_process,
            ):
                self._add_valid_doc(doc)
        else:
            if n_process != 1:
                LOGGER.warning("`n_process` is not available with spacy < 2.2.2")
            for doc in self.spacy_lang.pipe(
                texts, as_tuples=False, batch_size=batch_size,
            ):
                self._add_valid_doc(doc)

    def add_record(self, record: types.Record) -> None:
        """
        Add one record to the corpus, processing it into a :class:`spacy.tokens.Doc`
        using the :attr:`Corpus.spacy_lang` pipeline.

        Args:
            record
        """
        doc = self.spacy_lang(record[0])
        doc._.meta = record[1]
        self._add_valid_doc(doc)

    def add_records(
        self,
        records: Iterable[types.Record],
        batch_size: int = 1000,
        n_process: int = 1,
    ) -> None:
        """
        Add a stream of records to the corpus, efficiently processing them into
        :class:`spacy.tokens.Doc` s using the :attr:`Corpus.spacy_lang` pipeline.

        Args:
            records: Sequence of records to process and add to corpus.
            batch_size: Number of texts to buffer when processing with spaCy.
            n_process: Number of parallel processors to run when processing.
                If -1, this is set to ``multiprocessing.cpu_count()``.

                .. note:: This feature is only available in spaCy 2.2.2+.
        """
        if spacy.__version__ >= "2.2.2":
            for doc, meta in self.spacy_lang.pipe(
                records, as_tuples=True, batch_size=batch_size, n_process=n_process,
            ):
                doc._.meta = meta
                self._add_valid_doc(doc)
        else:
            if n_process != 1:
                LOGGER.warning("`n_process` is not available with spacy < 2.2.2")
            for doc, meta in self.spacy_lang.pipe(
                records, as_tuples=True, batch_size=batch_size,
            ):
                doc._.meta = meta
                self._add_valid_doc(doc)

    def add_doc(self, doc: Doc) -> None:
        """
        Add one :class:`spacy.tokens.Doc` to the corpus, provided it was processed
        using the :attr:`Corpus.spacy_lang` pipeline.

        Args:
            doc
        """
        if not isinstance(doc, Doc):
            raise TypeError(errors.type_invalid_msg("doc", type(doc), Doc))
        if doc.vocab is not self.spacy_lang.vocab:
            raise ValueError(
                f"doc.vocab ({doc.vocab}) must be the same as "
                f"corpus.vocab ({self.spacy_lang.vocab})"
            )
        self._add_valid_doc(doc)

    def add_docs(self, docs: Iterable[Doc]) -> None:
        """
        Add a stream of :class:`spacy.tokens.Doc` s to the corpus, provided
        they were processed using the :attr:`Corpus.spacy_lang` pipeline.

        Args:
            docs
        """
        for doc in docs:
            self.add_doc(doc)

    def _add_valid_doc(self, doc: Doc) -> None:
        self.docs.append(doc)
        self._doc_ids.append(id(doc))
        self.n_docs += 1
        self.n_tokens += len(doc)
        if doc.has_annotation("SENT_START"):
            self.n_sents += itertoolz.count(doc.sents)

    # get documents

    def get(
        self, match_func: Callable[[Doc], bool], limit: Optional[int] = None,
    ) -> Iterator[Doc]:
        """
        Get all (or N <= ``limit``) docs in :class:`Corpus` for which
        ``match_func(doc)`` is True.

        Args:
            match_func: Function that takes a :class:`spacy.tokens.Doc`
                as input and returns a boolean value. For example::

                    Corpus.get(lambda x: len(x) >= 100)

                gets all docs with at least 100 tokens. And::

                    Corpus.get(lambda doc: doc._.meta["author"] == "Burton DeWilde")

                gets all docs whose author was given as 'Burton DeWilde'.
            limit: Maximum number of matched docs to return.

        Yields:
            :class:`spacy.tokens.Doc`: Next document passing ``match_func``.

        .. tip:: To get doc(s) by index, treat :class:`Corpus` as a list and use
           Python's usual indexing and slicing: ``Corpus[0]`` gets the first
           document in the corpus; ``Corpus[:5]`` gets the first 5; etc.
        """
        matched_docs = (doc for doc in self if match_func(doc) is True)
        for doc in itertools.islice(matched_docs, limit):
            yield doc

    # remove documents

    def remove(
        self, match_func: Callable[[Doc], bool], limit: Optional[int] = None,
    ) -> None:
        """
        Remove all (or N <= ``limit``) docs in :class:`Corpus` for which
        ``match_func(doc)`` is True. Corpus doc/sent/token counts are adjusted
        accordingly.

        Args:
            match_func: Function that takes a :class:`spacy.tokens.Doc`
                and returns a boolean value. For example::

                    Corpus.remove(lambda x: len(x) >= 100)

                removes docs with at least 100 tokens. And::

                    Corpus.remove(lambda doc: doc._.meta["author"] == "Burton DeWilde")

                removes docs whose author was given as "Burton DeWilde".
            limit: Maximum number of matched docs to remove.

        .. tip:: To remove doc(s) by index, treat :class:`Corpus` as a list and use
           Python's usual indexing and slicing: ``del Corpus[0]`` removes the
           first document in the corpus; ``del Corpus[:5]`` removes the first
           5; etc.
        """
        matched_docs = (doc for doc in self if match_func(doc) is True)
        self._remove_many_docs_by_index(
            self._doc_ids.index(id(doc)) for doc in itertools.islice(matched_docs, limit)
        )

    def _remove_many_docs_by_index(self, idxs: Iterable[int]) -> None:
        for idx in sorted(idxs, reverse=True):
            self._remove_one_doc_by_index(idx)

    def _remove_one_doc_by_index(self, idx: int) -> None:
        doc = self.docs[idx]
        self.n_docs -= 1
        self.n_tokens -= len(doc)
        if doc.has_annotation("SENT_START"):
            self.n_sents -= itertoolz.count(doc.sents)
        del self.docs[idx]
        del self._doc_ids[idx]

    # useful properties

    @property
    def vectors(self) -> np.ndarray:
        """Constituent docs' word vectors stacked in a 2d array."""
        return np.vstack([doc.vector for doc in self])

    @property
    def vector_norms(self) -> np.ndarray:
        """Constituent docs' L2-normalized word vectors stacked in a 2d array."""
        return np.vstack([doc.vector_norm for doc in self])

    # useful methods

    def word_counts(
        self,
        *,
        by: str = "lemma",  # Literal["lemma", "lemma_", "lower", "lower_", "norm", "norm_", "orth", "orth_"]
        weighting: str = "count",  # Literal["count", "freq"]
        **kwargs,
    ) -> Dict[int, int | float] | Dict[str, int | float]:
        """
        Map the set of unique words in :class:`Corpus` to their counts as
        absolute, relative, or binary frequencies of occurence, similar to
        :meth:`Doc._.to_bag_of_words() <textacy.extensions.to_bag_of_words>`
        but aggregated over all docs.

        Args:
            by: Attribute by which spaCy ``Token`` s are grouped before counting,
                as given by ``getattr(token, by)``.
                If "lemma", tokens are grouped by their base form w/o inflections;
                if "lower", by the lowercase form of the token text;
                if "norm", by the normalized form of the token text;
                if "orth", by the token text exactly as it appears in documents.
                To output keys as strings, simply append an underscore to any of these;
                for example, "lemma_" creates a bag whose keys are token lemmas as strings.
            weighting: Type of weighting to assign to unique words given by ``by``.
                If "count", weights are the absolute number of occurrences (i.e. counts);
                if "freq", weights are counts normalized by the total token count,
                giving their relative frequency of occurrence.
            **kwargs: Passed directly on to :func:`textacy.extract.words()`
                - filter_stops: If True, stop words are removed before counting.
                - filter_punct: If True, punctuation tokens are removed before counting.
                - filter_nums: If True, number-like tokens are removed before counting.

        Returns:
            Mapping of a unique word id or string (depending on the value of ``by``)
            to its absolute, relative, or binary frequency of occurrence
            (depending on the value of ``weighting``).

        See Also:
            :func:`textacy.vsm.get_term_freqs() <textacy.vsm.matrix_utils.get_term_freqs>`
        """
        word_counts_: Union[Counter[Any], Dict[Any, Union[int, float]]]
        word_counts_ = collections.Counter()
        for doc in self:
            word_counts_.update(
                extensions.to_bag_of_words(doc, by=by, weighting="count", **kwargs)
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
                errors.value_invalid_msg("weighting", weighting, {"count", "freq"})
            )
        return word_counts_

    def word_doc_counts(
        self,
        *,
        by: str = "lemma",  # Literal["lemma", "lemma_", "lower", "lower_", "norm", "norm_", "orth", "orth_"]
        weighting: str = "count",  # Literal["count", "freq", "idf"]
        smooth_idf: bool = True,
        **kwargs
    ) -> Dict[int, int | float] | Dict[str, int | float]:
        """
        Map the set of unique words in :class:`Corpus` to their *document* counts
        as absolute, relative, or inverse frequencies of occurence.

        Args:
            by: Attribute by which spaCy ``Token`` s are grouped before counting,
                as given by ``getattr(token, by)``.
                If "lemma", tokens are grouped by their base form w/o inflections;
                if "lower", by the lowercase form of the token text;
                if "norm", by the normalized form of the token text;
                if "orth", by the token text exactly as it appears in documents.
                To output keys as strings, simply append an underscore to any of these;
                for example, "lemma_" creates a bag whose keys are token lemmas as strings.
            weighting: Type of weighting to assign to unique words given by ``by``.
                If "count", weights are the absolute number of occurrences (i.e. counts);
                if "freq", weights are counts normalized by the total token count,
                giving their relative frequency of occurrence;
                if "idf", weights are the log of the inverse relative frequencies, i.e.
                ``log(n_docs / word_doc_count)`` or, if ``smooth_idf`` is True,
                ``log(1 + (n_docs / word_doc_count))``.
            smooth_idf: If True, add 1 to all word doc counts when
                calculating "idf" weighting, equivalent to adding a single
                document to the corpus containing every unique word.

        Returns:
            Mapping of a unique word id or string (depending on the value of ``by``)
            to the number of documents in which it appears,
            weighted as absolute, relative, or inverse frequency of occurrence
            (depending on the value of ``weighting``).

        See Also:
            :func:`textacy.vsm.get_doc_freqs() <textacy.vsm.matrix_utils.get_doc_freqs>`
        """
        word_doc_counts_: Union[Counter[Any], Dict[Any, Union[int, float]]]
        word_doc_counts_ = collections.Counter()
        for doc in self:
            word_doc_counts_.update(
                extensions.to_bag_of_words(doc, by=by, weighting="binary", **kwargs)
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
                    word: math.log1p(n_docs / count)
                    for word, count in word_doc_counts_.items()
                }
            else:
                word_doc_counts_ = {
                    word: math.log(n_docs / count)
                    for word, count in word_doc_counts_.items()
                }
        else:
            raise ValueError(
                errors.value_invalid_msg(
                    "weighting", weighting, {"count", "freq", "idf"}
                )
            )
        return word_doc_counts_

    def agg_metadata(
        self,
        name: str,
        agg_func: Callable[[Iterable[Any]], Any],
        default: Optional[Any] = None,
    ) -> Any:
        """
        Aggregate values for a particular metadata field over all documents
        in :class:`Corpus`.

        Args:
            name: Name of metadata field (key) in :class:`Doc._.meta`.
            agg_func: Callable that accepts an iterable of field values
                and outputs a single, aggregated result.
            default: Default field value to use if ``name`` is not found
                in a given document's metadata.

        Returns:
            Aggregated value for metadata field.
        """
        return agg_func(doc._.meta.get(name, default) for doc in self)

    # file io

    def save(self, filepath: types.PathLike, store_user_data: bool = True):
        """
        Save :class:`Corpus` to disk as binary data.

        Args:
            filepath: Full path to file on disk where :class:`Corpus` data
                will be saved as a binary file.
            store_user_data: If True, store user data and values of
                custom extension attributes along with core spaCy attributes.

        See Also:
            - :meth:`Corpus.load()`
            - :class:`spacy.tokens.DocBin`
        """
        attrs = [
            spacy.attrs.ORTH,
            spacy.attrs.SPACY,
        ]
        if self[0].has_annotation("TAG"):
            attrs.append(spacy.attrs.TAG)
        if self[0].has_annotation("DEP"):
            attrs.append(spacy.attrs.HEAD)
            attrs.append(spacy.attrs.DEP)
        # NOTE: HEAD sets sentence boundaries implicitly based on tree structure, so
        # also setting SENT_START would potentially conflict with existing annotations.
        elif self[0].has_annotation("SENT_START"):
            attrs.append(spacy.attrs.SENT_START)
        if self[0].has_annotation("ENT_IOB"):
            attrs.append(spacy.attrs.ENT_IOB)
            attrs.append(spacy.attrs.ENT_TYPE)
        doc_bin = DocBin(attrs=attrs, store_user_data=store_user_data)
        for doc in self:
            doc_bin.add(doc)
        with tio.open_sesame(filepath, mode="wb") as f:
            f.write(doc_bin.to_bytes())

    @classmethod
    def load(
        cls,
        lang: types.LangLike,
        filepath: types.PathLike,
        store_user_data: bool = True,
    ) -> "Corpus":
        """
        Load previously saved :class:`Corpus` binary data, reproduce the original
        `:class:`spacy.tokens.Doc`s tokens and annotations, and instantiate
        a new :class:`Corpus` from them.

        Args:
            lang
            filepath: Full path to file on disk where :class:`Corpus` data
                was previously saved as a binary file.
            store_user_data: If True, load stored user data and values
                of custom extension attributes along with core spaCy attributes.

        Returns:
            :class:`Corpus`

        See Also:
            - :meth:`Corpus.save()`
            - :class:`spacy.tokens.DocBin`
        """
        spacy_lang = spacier.utils.resolve_langlike(lang)
        with tio.open_sesame(filepath, mode="rb") as f:
            bytes_data = f.read()
        doc_bin = DocBin(store_user_data=store_user_data).from_bytes(bytes_data)
        return cls(spacy_lang, data=doc_bin.get_docs(spacy_lang.vocab))
