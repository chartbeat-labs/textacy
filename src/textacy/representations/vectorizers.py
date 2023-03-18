# mypy: ignore-errors
"""
Vectorizers
-----------

:mod:`textacy.representations.vectorizers`:  Transform a collection of tokenized docs
into a doc-term matrix of shape (# docs, # unique terms), with various ways to filter
or limit included terms and flexible weighting schemes for their values.

A second option aggregates terms in tokenized documents by provided group labels,
resulting in a "group-term-matrix" of shape (# unique groups, # unique terms),
with filtering and weighting functionality as described above.

See the :class:`Vectorizer` and :class:`GroupVectorizer` docstrings for usage
examples and explanations of the various weighting schemes.
"""
from __future__ import annotations

import collections
import collections.abc
import operator
from array import array
from typing import DefaultDict, Iterable, Literal, Optional, Union

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize as normalize_mat

from .. import errors
from .matrix_utils import filter_terms_by_df, get_doc_lengths, get_inverse_doc_freqs


BM25_K1 = 1.6  # value typically bounded in [1.2, 2.0]
BM25_B = 0.75


class Vectorizer:
    """
    Transform one or more tokenized documents into a sparse document-term matrix
    of shape (# docs, # unique terms), with flexible weighting/normalization of values.

    Stream a corpus with metadata from disk:

    .. code-block:: pycon

        >>> ds = textacy.datasets.CapitolWords()
        >>> records = ds.records(limit=1000)
        >>> corpus = textacy.Corpus("en_core_web_sm", data=records)
        >>> print(corpus)
        Corpus(1000 docs, 538397 tokens)

    Tokenize and vectorize the first 600 documents of this corpus:

    .. code-block:: pycon

        >>> tokenized_docs = (
        ...     (term.lemma_ for term in textacy.extract.terms(doc, ngs=1, ents=True))
        ...     for doc in corpus[:600])
        >>> vectorizer = Vectorizer(
        ...     tf_type="linear", idf_type="smooth", norm="l2",
        ...     min_df=3, max_df=0.95)
        >>> doc_term_matrix = vectorizer.fit_transform(tokenized_docs)
        >>> doc_term_matrix
        <600x4412 sparse matrix of type '<class 'numpy.float64'>'
                with 65210 stored elements in Compressed Sparse Row format>

    Tokenize and vectorize the remaining 400 documents of the corpus, using only
    the groups, terms, and weights learned in the previous step:

    .. code-block:: pycon

        >>> tokenized_docs = (
        ...     (term.lemma_ for term in textacy.extract.terms(doc, ngs=1, ents=True))
        ...     for doc in corpus[600:])
        >>> doc_term_matrix = vectorizer.transform(tokenized_docs)
        >>> doc_term_matrix
        <400x4412 sparse matrix of type '<class 'numpy.float64'>'
                with 36212 stored elements in Compressed Sparse Row format>

    Inspect the terms associated with columns; they're sorted alphabetically:

    .. code-block:: pycon

        >>> vectorizer.terms_list[:5]
        ['', '$', '$ 1 million', '$ 1.2 billion', '$ 10 billion']

    (Btw: That empty string shouldn't be there. Somehow, spaCy is labeling it as
    a named entity...?)

    If known in advance, limit the terms included in vectorized outputs
    to a particular set of values:

    .. code-block:: pycon

        >>> tokenized_docs = (
        ...     (term.lemma_ for term in textacy.extract.terms(doc, ngs=1, ents=True))
        ...     for doc in corpus[:600])
        >>> vectorizer = Vectorizer(
        ...     idf_type="smooth", norm="l2",
        ...     min_df=3, max_df=0.95,
        ...     vocabulary_terms=["president", "bill", "unanimous", "distinguished", "american"])
        >>> doc_term_matrix = vectorizer.fit_transform(tokenized_docs)
        >>> doc_term_matrix
        <600x5 sparse matrix of type '<class 'numpy.float64'>'
                with 516 stored elements in Compressed Sparse Row format>
        >>> vectorizer.terms_list
        ['american', 'bill', 'distinguished', 'president', 'unanimous']

    Specify different weighting schemes to determine values in the matrix,
    adding or customizing individual components, as desired:

    .. code-block:: pycon

        >>> tokenized_docs = [
            [term.lemma_ for term in textacy.extract.terms(doc, ngs=1, ents=True)]
            for doc in corpus[:600]]
        >>> doc_term_matrix = Vectorizer(
        ...     tf_type="linear", norm=None, min_df=3, max_df=0.95
        ... ).fit_transform(tokenized_docs)
        >>> print(doc_term_matrix[:8, vectorizer.vocabulary_terms["$"]].toarray())
        [[0]
         [0]
         [1]
         [4]
         [0]
         [0]
         [2]
         [4]]
        >>> doc_term_matrix = Vectorizer(
        ...     tf_type="sqrt", dl_type="sqrt", norm=None, min_df=3, max_df=0.95
        ... ).fit_transform(tokenized_docs)
        >>> print(doc_term_matrix[:8, vectorizer.vocabulary_terms["$"]].toarray())
        [[0.        ]
         [0.        ]
         [0.10660036]
         [0.2773501 ]
         [0.        ]
         [0.        ]
         [0.11704115]
         [0.24806947]]
        >>> doc_term_matrix = Vectorizer(
        ...     tf_type="bm25", idf_type="smooth", norm=None, min_df=3, max_df=0.95
        ... ).fit_transform(tokenized_docs)
        >>> print(doc_term_matrix[:8, vectorizer.vocabulary_terms["$"]].toarray())
        [[0.        ]
         [0.        ]
         [2.68009606]
         [4.97732126]
         [0.        ]
         [0.        ]
         [3.87124987]
         [4.97732126]]

    If you're not sure what's going on mathematically, :attr:`Vectorizer.weighting`
    gives the formula being used to calculate weights, based on the parameters
    set when initializing the vectorizer:

    .. code-block:: pycon

        >>> vectorizer.weighting
        '(tf * (k + 1)) / (k + tf) * log((n_docs + 1) / (df + 1)) + 1'

    In general, weights may consist of a local component (term frequency),
    a global component (inverse document frequency), and a normalization
    component (document length). Individual components may be modified:
    they may have different scaling (e.g. tf vs. sqrt(tf)) or different behaviors
    (e.g. "standard" idf vs bm25's version). There are *many* possible weightings,
    and some may be better for particular use cases than others. When in doubt,
    though, just go with something standard.

    - "tf": Weights are simply the absolute per-document term frequencies (tfs),
      i.e. value (i, j) in an output doc-term matrix corresponds to the number
      of occurrences of term j in doc i. Terms appearing many times in a given
      doc receive higher weights than less common terms.
      Params: ``tf_type="linear", apply_idf=False, apply_dl=False``
    - "tfidf": Doc-specific, *local* tfs are multiplied by their corpus-wide,
      *global* inverse document frequencies (idfs). Terms appearing in many docs
      have higher document frequencies (dfs), correspondingly smaller idfs, and
      in turn, lower weights.
      Params: ``tf_type="linear", apply_idf=True, idf_type="smooth", apply_dl=False``
    - "bm25": This scheme includes a local tf component that increases asymptotically,
      so higher tfs have diminishing effects on the overall weight; a global idf
      component that can go *negative* for terms that appear in a sufficiently
      high proportion of docs; as well as a row-wise normalization that accounts for
      document length, such that terms in shorter docs hit the tf asymptote sooner
      than those in longer docs.
      Params: ``tf_type="bm25", apply_idf=True, idf_type="bm25", apply_dl=True``
    - "binary": This weighting scheme simply replaces all non-zero tfs with 1,
      indicating the presence or absence of a term in a particular doc. That's it.
      Params: ``tf_type="binary", apply_idf=False, apply_dl=False``

    Slightly altered versions of these "standard" weighting schemes are common,
    and may have better behavior in general use cases:

    - "lucene-style tfidf": Adds a doc-length normalization to the usual local
      and global components.
      Params: ``tf_type="linear", apply_idf=True, idf_type="smooth", apply_dl=True, dl_type="sqrt"``
    - "lucene-style bm25": Uses a smoothed idf instead of the classic bm25 variant
      to prevent weights on terms from going negative.
      Params: ``tf_type="bm25", apply_idf=True, idf_type="smooth", apply_dl=True, dl_type="linear"``

    Args:
        tf_type: Type of term frequency (tf) to use for weights' local component:

            - "linear": tf (tfs are already linear, so left as-is)
            - "sqrt": tf => sqrt(tf)
            - "log": tf => log(tf) + 1
            - "binary": tf => 1

        idf_type: Type of inverse document frequency (idf) to use for weights'
            global component:

            - "standard": idf = log(n_docs / df) + 1.0
            - "smooth": idf = log(n_docs + 1 / df + 1) + 1.0, i.e. 1 is added
              to all document frequencies, as if a single document containing
              every unique term was added to the corpus.
            - "bm25": idf = log((n_docs - df + 0.5) / (df + 0.5)), which is
              a form commonly used in information retrieval that allows for
              very common terms to receive negative weights.
            - None: no global weighting is applied to local term weights.

        dl_type: Type of document-length scaling to use for weights'
            normalization component:

            - "linear": dl (dls are already linear, so left as-is)
            - "sqrt": dl => sqrt(dl)
            - "log": dl => log(dl)
            - None: no normalization is applied to local (+global?) weights

        norm: If "l1" or "l2", normalize weights by the L1 or L2 norms, respectively,
            of row-wise vectors; otherwise, don't.
        min_df: Minimum number of documents in which a term must appear for it to be
            included in the vocabulary and as a column in a transformed doc-term matrix.
            If float, value is the fractional proportion of the total number of docs,
            which must be in [0.0, 1.0]; if int, value is the absolute number.
        max_df: Maximum number of documents in which a term may appear for it to be
            included in the vocabulary and as a column in a transformed doc-term matrix.
            If float, value is the fractional proportion of the total number of docs,
            which must be in [0.0, 1.0]; if int, value is the absolute number.
        max_n_terms: If specified, only include terms whose document frequency is within
            the top ``max_n_terms``.
        vocabulary_terms: Mapping of unique term string to unique term id, or
            an iterable of term strings that gets converted into such a mapping.
            Note that, if specified, vectorized outputs will include *only* these terms.

    Attributes:
        vocabulary_terms (dict[str, int]): Mapping of unique term string to unique
            term id, either provided on instantiation or generated by calling
            :meth:`Vectorizer.fit()` on a collection of tokenized documents.
    """

    def __init__(
        self,
        *,
        tf_type: Literal["linear", "sqrt", "log", "binary"] = "linear",
        idf_type: Optional[Literal["standard", "smooth", "bm25"]] = None,
        dl_type: Optional[Literal["linear", "sqrt", "log"]] = None,
        norm: Optional[Literal["l1", "l2"]] = None,
        min_df: int | float = 1,
        max_df: int | float = 1.0,
        max_n_terms: Optional[int] = None,
        vocabulary_terms: Optional[dict[str, int] | Iterable[str]] = None,
    ):
        # sanity check numeric arguments
        if min_df < 0 or max_df < 0:
            raise ValueError("`min_df` and `max_df` must be positive numbers or None")
        if max_n_terms and max_n_terms < 0:
            raise ValueError("`max_n_terms` must be a positive integer or None")
        self.tf_type = tf_type
        self.idf_type = idf_type
        self.dl_type = dl_type
        self.norm = norm
        self.min_df = min_df
        self.max_df = max_df
        self.max_n_terms = max_n_terms
        self.vocabulary_terms, self._fixed_terms = self._validate_vocabulary(
            vocabulary_terms
        )
        self.id_to_term_: dict[int, str] = {}
        self._idf_diag = None
        self._avg_doc_length = None

    def _validate_vocabulary(
        self, vocabulary: dict[str, int] | Iterable[str]
    ) -> tuple[dict[str, int], bool]:
        """
        Validate an input vocabulary. If it's a mapping, ensure that term ids
        are unique and compact (i.e. without any gaps between 0 and the number
        of terms in ``vocabulary``. If it's a sequence, sort terms then assign
        integer ids in ascending order.
        """
        if vocabulary is not None:
            if not isinstance(vocabulary, collections.abc.Mapping):
                vocab: dict[str, int] = {}
                for i, term in enumerate(sorted(vocabulary)):
                    if vocab.setdefault(term, i) != i:
                        raise ValueError(
                            f"Terms in `vocabulary` must be unique, but '{term}' "
                            "was found more than once."
                        )
                vocabulary = vocab
            else:
                ids = set(vocabulary.values())
                if len(ids) != len(vocabulary):
                    counts = collections.Counter(vocabulary.values())
                    n_dupe_term_ids = sum(
                        1
                        for term_id, term_id_count in counts.items()
                        if term_id_count > 1
                    )
                    raise ValueError(
                        "Term ids in `vocabulary` must be unique, but "
                        f"{n_dupe_term_ids} ids were assigned to more than one term."
                    )
                for i in range(len(vocabulary)):
                    if i not in ids:
                        raise ValueError(
                            "Term ids in `vocabulary` must be compact, i.e. "
                            f"not have any gaps, but term id {i} is missing from "
                            f"a vocabulary of {len(vocabulary)} terms"
                        )
            if not vocabulary:
                raise ValueError("`vocabulary` must not be empty.")
            is_fixed = True
        else:
            is_fixed = False
        return (vocabulary, is_fixed)

    def _check_vocabulary(self):
        """
        Check that instance has a valid vocabulary mapping;
        if not, raise a ValueError.
        """
        if not isinstance(self.vocabulary_terms, collections.abc.Mapping):
            raise ValueError("vocabulary hasn't been built; call `Vectorizer.fit()`")
        if len(self.vocabulary_terms) == 0:
            raise ValueError("vocabulary is empty")

    @property
    def id_to_term(self) -> dict[int, str]:
        """
        Mapping of unique term id (int) to unique term string (str), i.e.
        the inverse of :attr:`Vectorizer.vocabulary`. This attribute is only
        generated if needed, and it is automatically kept in sync with the
        corresponding vocabulary.
        """
        if len(self.id_to_term_) != self.vocabulary_terms:
            self.id_to_term_ = {
                term_id: term_str for term_str, term_id in self.vocabulary_terms.items()
            }
        return self.id_to_term_

    # TODO: Do we *want* to allow setting to this property?
    # @id_to_term.setter
    # def id_to_term(self, new_id_to_term):
    #     self.id_to_term_ = new_id_to_term
    #     self.vocabulary_terms = {
    #         term_str: term_id for term_id, term_str in new_id_to_term.items()}

    @property
    def terms_list(self) -> list[str]:
        """
        List of term strings in column order of vectorized outputs. For example,
        ``terms_list[0]`` gives the term assigned to the first column in an
        output doc-term-matrix, ``doc_term_matrix[:, 0]``.
        """
        self._check_vocabulary()
        return [
            term_str
            for term_str, _ in sorted(
                self.vocabulary_terms.items(), key=operator.itemgetter(1)
            )
        ]

    def fit(self, tokenized_docs: Iterable[Iterable[str]]) -> "Vectorizer":
        """
        Count terms in ``tokenized_docs`` and, if not already provided, build up
        a vocabulary based those terms. Fit and store global weights (IDFs)
        and, if needed for term weighting, the average document length.

        Args:
            tokenized_docs: A sequence of tokenized documents, where each is
                a sequence of term strings. For example::

                    >>> ([tok.lemma_ for tok in spacy_doc]
                    ...  for spacy_doc in spacy_docs)
                    >>> ((ne.text for ne in extract.entities(doc))
                    ...  for doc in corpus)

        Returns:
            Vectorizer instance that has just been fit.
        """
        _ = self._fit(tokenized_docs)
        return self

    def fit_transform(self, tokenized_docs: Iterable[Iterable[str]]) -> sp.csr_matrix:
        """
        Count terms in ``tokenized_docs`` and, if not already provided, build up
        a vocabulary based those terms. Fit and store global weights (IDFs)
        and, if needed for term weighting, the average document length.
        Transform ``tokenized_docs`` into a document-term matrix with values
        weighted according to the parameters in :class:`Vectorizer` initialization.

        Args:
            tokenized_docs: A sequence of tokenized documents, where each is
                a sequence of term strings. For example::

                    >>> ([tok.lemma_ for tok in spacy_doc]
                    ...  for spacy_doc in spacy_docs)
                    >>> ((ne.text for ne in extract.entities(doc))
                    ...  for doc in corpus)

        Returns:
            The transformed document-term matrix, where rows correspond to documents
            and columns correspond to terms, as a sparse row matrix.
        """
        # count terms and fit global weights
        doc_term_matrix = self._fit(tokenized_docs)
        # re-weight values in doc-term matrix, as specified in init
        doc_term_matrix = self._reweight_values(doc_term_matrix)
        return doc_term_matrix

    def transform(self, tokenized_docs: Iterable[Iterable[str]]) -> sp.csr_matrix:
        """
        Transform ``tokenized_docs`` into a document-term matrix with values
        weighted according to the parameters in :class:`Vectorizer` initialization
        and the global weights computed by calling :meth:`Vectorizer.fit()`.

        Args:
            tokenized_docs: A sequence of tokenized documents, where each is
                a sequence of term strings. For example::

                    >>> ([tok.lemma_ for tok in spacy_doc]
                    ...  for spacy_doc in spacy_docs)
                    >>> ((ne.text for ne in extract.entities(doc))
                    ...  for doc in corpus)

        Returns:
            The transformed document-term matrix, where rows correspond to documents
            and columns correspond to terms, as a sparse row matrix.

        Note:
            For best results, the tokenization used to produce ``tokenized_docs``
            should be the same as was applied to the docs used in fitting this
            vectorizer or in generating a fixed input vocabulary.

            Consider an extreme case where the docs used in fitting consist of
            lowercased (non-numeric) terms, while the docs to be transformed are
            all uppercased: The output doc-term-matrix will be empty.
        """
        self._check_vocabulary()
        doc_term_matrix, _ = self._count_terms(tokenized_docs, True)
        return self._reweight_values(doc_term_matrix)

    def _fit(self, tokenized_docs: Iterable[Iterable[str]]) -> sp.csr_matrix:
        """
        Count terms and, if :attr:`Vectorizer.fixed_terms` is False, build up
        a vocabulary based on the terms found in ``tokenized_docs``. Transform
        ``tokenized_docs`` into a document-term matrix with absolute tf weights.
        Store global weights (IDFs) and, if :attr:`Vectorizer.doc_length_norm`
        is not None, the average doc length.

        Args:
            tokenized_docs

        Returns:
            Document-term matrix.
        """
        # count terms and, if not provided on init, build up a vocabulary
        doc_term_matrix, vocabulary_terms = self._count_terms(
            tokenized_docs, self._fixed_terms
        )

        if self._fixed_terms is False:
            # filter terms by doc freq or info content, as specified in init
            doc_term_matrix, vocabulary_terms = self._filter_terms(
                doc_term_matrix, vocabulary_terms
            )
            # sort features alphabetically (vocabulary_terms modified in-place)
            doc_term_matrix = self._sort_vocab_and_matrix(
                doc_term_matrix, vocabulary_terms, axis="columns"
            )
            # *now* vocabulary_terms are known and fixed
            self.vocabulary_terms = vocabulary_terms
            self._fixed_terms = True

        n_docs, n_terms = doc_term_matrix.shape

        if self.idf_type:
            # store the global weights as a diagonal sparse matrix of idfs
            idfs = get_inverse_doc_freqs(doc_term_matrix, type_=self.idf_type)
            self._idf_diag = sp.spdiags(
                idfs, diags=0, m=n_terms, n=n_terms, format="csr"
            )

        if self.tf_type == "bm25" and self.dl_type:
            # store the avg document length, used in bm25 weighting to normalize
            # term weights by the length of the containing documents
            self._avg_doc_length = get_doc_lengths(
                doc_term_matrix, type_=self.dl_type
            ).mean()

        return doc_term_matrix

    def _count_terms(
        self, tokenized_docs: Iterable[Iterable[str]], fixed_vocab: bool
    ) -> tuple[sp.csr_matrix, dict[str, int]]:
        """
        Count terms found in ``tokenized_docs`` and, if ``fixed_vocab`` is False,
        build up a vocabulary based on those terms.

        Args:
            tokenized_docs
            fixed_vocab

        Returns:
            Document-term matrix and vocabulary used to make it.
        """
        if fixed_vocab is False:
            # add a new value when a new term is seen
            vocabulary = collections.defaultdict()
            vocabulary.default_factory = vocabulary.__len__
        else:
            vocabulary = self.vocabulary_terms

        indices = array(str("i"))
        indptr = array(str("i"), [0])
        for terms in tokenized_docs:
            for term in terms:
                try:
                    indices.append(vocabulary[term])
                except KeyError:
                    # ignore out-of-vocabulary terms when _fixed_terms=True
                    continue
            indptr.append(len(indices))

        if fixed_vocab is False:
            # we no longer want defaultdict behaviour
            vocabulary = dict(vocabulary)

        indices = np.frombuffer(indices, dtype=np.intc)
        indptr = np.frombuffer(indptr, dtype=np.intc)
        data = np.ones(len(indices))

        # build the matrix, then consolidate duplicate entries
        # by adding them together, in-place
        doc_term_matrix = sp.csr_matrix(
            (data, indices, indptr),
            shape=(len(indptr) - 1, len(vocabulary)),
            dtype=np.int32,
        )
        doc_term_matrix.sum_duplicates()

        # pretty sure this is a good thing to do... o_O
        doc_term_matrix.sort_indices()

        return doc_term_matrix, vocabulary

    def _filter_terms(
        self, doc_term_matrix: sp.csr_matrix, vocabulary: dict[str, int]
    ) -> tuple[sp.csr_matrix, dict[str, int]]:
        """
        Filter terms in ``vocabulary`` by their document frequency or information
        content, as specified in :class:`Vectorizer` initialization.

        Args:
            doc_term_matrix
            vocabulary

        Returns:
            Filtered document-term matrix and filtered vocabulary to go with it.
        """
        if self.max_df != 1.0 or self.min_df != 1 or self.max_n_terms is not None:
            doc_term_matrix, vocabulary = filter_terms_by_df(
                doc_term_matrix,
                vocabulary,
                max_df=self.max_df,
                min_df=self.min_df,
                max_n_terms=self.max_n_terms,
            )
        return doc_term_matrix, vocabulary

    def _sort_vocab_and_matrix(
        self,
        matrix: sp.csr_matrix,
        vocabulary: dict[str, int],
        axis: Literal["rows", 0] | Literal["columns", 1],
    ) -> sp.csr_matrix:
        """
        Sort terms in ``vocabulary`` alphabetically, modifying the vocabulary
        in-place, and returning a correspondingly reordered ``matrix`` along
        its rows or columns, depending on ``axis``.

        Args:
            matrix
            vocabulary
            axis
        """
        sorted_vocab = sorted(vocabulary.items())
        new_idx_array = np.empty(len(sorted_vocab), dtype=np.int32)
        for new_idx, (term, old_idx) in enumerate(sorted_vocab):
            new_idx_array[new_idx] = old_idx
            vocabulary[term] = new_idx
        # use fancy indexing to reorder rows or columns
        if axis == "rows" or axis == 0:
            return matrix[new_idx_array, :]
        elif axis == "columns" or axis == 1:
            return matrix[:, new_idx_array]
        else:
            raise ValueError(
                errors.value_invalid_msg("axis", axis, {"rows", "columns", 0, 1})
            )

    def _reweight_values(self, doc_term_matrix: sp.csr_matrix) -> sp.csr_matrix:
        """
        Re-weight values in a doc-term matrix according to parameters specified
        in :class:`Vectorizer` initialization: binary or tf-idf weighting,
        sublinear term-frequency, document-normalized weights.

        Args:
            doc_term_matrix

        Returns:
            Reweighted doc-term matrix.
        """
        # re-weight the local components (term freqs)
        if self.tf_type == "binary":
            doc_term_matrix.data.fill(1)
        elif self.tf_type == "bm25":
            if not self.dl_type:
                doc_term_matrix.data = (
                    doc_term_matrix.data
                    * (BM25_K1 + 1.0)
                    / (BM25_K1 + doc_term_matrix.data)
                )
            else:
                dls = get_doc_lengths(doc_term_matrix, type_=self.dl_type)
                length_norm = (1 - BM25_B) + (BM25_B * (dls / self._avg_doc_length))
                doc_term_matrix = doc_term_matrix.tocoo(copy=False)
                doc_term_matrix.data = (
                    doc_term_matrix.data
                    * (BM25_K1 + 1.0)
                    / (
                        doc_term_matrix.data
                        + (BM25_K1 * length_norm[doc_term_matrix.row])
                    )
                )
                doc_term_matrix = doc_term_matrix.tocsr(copy=False)
        elif self.tf_type == "sqrt":
            _ = np.sqrt(doc_term_matrix.data, doc_term_matrix.data, casting="unsafe")
        elif self.tf_type == "log":
            _ = np.log(doc_term_matrix.data, doc_term_matrix.data, casting="unsafe")
            doc_term_matrix.data += 1.0
        elif self.tf_type == "linear":
            pass  # tfs are already linear
        else:
            # this should never raise, i'm just being a worrywart
            raise ValueError(
                errors.value_invalid_msg(
                    "tf_type", self.tf_type, {"binary", "bm25", "sqrt", "log", "linear"}
                )
            )

        # apply the global component (idfs), column-wise
        if self.idf_type:
            doc_term_matrix = doc_term_matrix * self._idf_diag

        # apply normalizations, row-wise
        # unless we've already handled it for bm25-style tf
        if self.dl_type and self.tf_type != "bm25":
            n_docs, _ = doc_term_matrix.shape
            dls = get_doc_lengths(doc_term_matrix, type_=self.dl_type)
            dl_diag = sp.spdiags(1.0 / dls, diags=0, m=n_docs, n=n_docs, format="csr")
            doc_term_matrix = dl_diag * doc_term_matrix
        if self.norm is not None:
            doc_term_matrix = normalize_mat(
                doc_term_matrix, norm=self.norm, axis=1, copy=False
            )

        return doc_term_matrix

    @property
    def weighting(self) -> str:
        """
        A mathematical representation of the overall weighting scheme
        used to determine values in the vectorized matrix, depending on the
        params used to initialize the :class:`Vectorizer`.
        """
        w = []
        tf_types = {
            "binary": "1",
            "linear": "tf",
            "sqrt": "sqrt(tf)",
            "log": "log(tf)",
            "bm25": {
                True: "(tf * (k + 1)) / (tf + k * (1 - b + b * (length / avg(lengths)))",
                False: "(tf * (k + 1)) / (tf + k)",
            },
        }
        idf_types = {
            "standard": "log(n_docs / df) + 1",
            "smooth": "log((n_docs + 1) / (df + 1)) + 1",
            "bm25": "log((n_docs - df + 0.5) / (df + 0.5))",
        }
        dl_types = {
            "linear": "1/length",
            "sqrt": "1/sqrt(length)",
            "log": "1/log(length) + 1",
        }
        if self.tf_type == "bm25":
            w.append(tf_types[self.tf_type][bool(self.dl_type)])
        else:
            w.append(tf_types[self.tf_type])
        if self.idf_type:
            w.append(idf_types[self.idf_type])
        if self.dl_type and self.tf_type != "bm25":
            w.append(dl_types[self.dl_type])
        return " * ".join(w)


class GroupVectorizer(Vectorizer):
    """
    Transform one or more tokenized documents into a group-term matrix of
    shape (# groups, # unique terms), with tf-, tf-idf, or binary-weighted values.

    This is an extension of typical document-term matrix vectorization, where
    terms are grouped by the documents in which they co-occur. It allows for
    customized grouping, such as by a shared author or publication year, that
    may span multiple documents, without forcing users to merge those documents
    themselves.

    Stream a corpus with metadata from disk::

        >>> ds = textacy.datasets.CapitolWords()
        >>> records = ds.records(limit=1000)
        >>> corpus = textacy.Corpus("en_core_web_sm", data=records)
        >>> corpus
        Corpus(1000 docs, 538397 tokens)

    Tokenize and vectorize the first 600 documents of this corpus, where terms
    are grouped not by documents but by a categorical value in the docs' metadata::

        >>> tokenized_docs, groups = textacy.io.unzip(
        ...     ((term.lemma_ for term in textacy.extract.terms(doc, ngs=1, ents=True)),
        ...      doc._.meta["speaker_name"])
        ...     for doc in corpus[:600])
        >>> vectorizer = GroupVectorizer(
        ...     tf_type="linear", idf_type="smooth", norm="l2",
        ...     min_df=3, max_df=0.95)
        >>> grp_term_matrix = vectorizer.fit_transform(tokenized_docs, groups)
        >>> grp_term_matrix
        <5x1822 sparse matrix of type '<class 'numpy.float64'>'
                with 6139 stored elements in Compressed Sparse Row format>

    Tokenize and vectorize the remaining 400 documents of the corpus, using only
    the groups, terms, and weights learned in the previous step::

        >>> tokenized_docs, groups = textacy.io.unzip(
        ...     ((term.lemma_ for term in textacy.extract.terms(doc, ngs=1, ents=True)),
        ...      doc._.meta["speaker_name"])
        ...     for doc in corpus[600:])
        >>> grp_term_matrix = vectorizer.transform(tokenized_docs, groups)
        >>> grp_term_matrix
        <5x1822 sparse matrix of type '<class 'numpy.float64'>'
                with 4414 stored elements in Compressed Sparse Row format>

    Inspect the terms associated with columns and groups associated with rows;
    they're sorted alphabetically::

        >>> vectorizer.terms_list[:5]
        ['', '$ 1 million', '$ 160 million', '$ 5 billion', '$ 7 billion']
        >>> vectorizer.grps_list
        ['Bernie Sanders', 'John Kasich', 'Joseph Biden', 'Lindsey Graham', 'Rick Santorum']

    If known in advance, limit the terms and/or groups included in vectorized outputs
    to a particular set of values::

        >>> tokenized_docs, groups = textacy.io.unzip(
        ...     ((term.lemma_ for term in textacy.extract.terms(doc, ngs=1, ents=True)),
        ...      doc._.meta["speaker_name"])
        ...     for doc in corpus[:600])
        >>> vectorizer = GroupVectorizer(
        ...     tf_type="linear", idf_type="smooth", norm="l2",
        ...     min_df=3, max_df=0.95,
        ...     vocabulary_terms=["legislation", "federal government", "house", "constitutional"],
        ...     vocabulary_grps=["Bernie Sanders", "Lindsey Graham", "Rick Santorum"])
        >>> grp_term_matrix = vectorizer.fit_transform(tokenized_docs, groups)
        >>> grp_term_matrix
        <3x4 sparse matrix of type '<class 'numpy.float64'>'
                with 9 stored elements in Compressed Sparse Row format>
        >>> vectorizer.terms_list
        ['constitutional', 'federal government', 'house', 'legislation']
        >>> vectorizer.grps_list
        ['Bernie Sanders', 'Lindsey Graham', 'Rick Santorum']

    For a discussion of the various weighting schemes that can be applied, check
    out the :class:`Vectorizer` docstring.

    Args:
        tf_type: Type of term frequency (tf) to use for weights' local component:

            - "linear": tf (tfs are already linear, so left as-is)
            - "sqrt": tf => sqrt(tf)
            - "log": tf => log(tf) + 1
            - "binary": tf => 1

        idf_type: Type of inverse document frequency (idf) to use for weights'
            global component:

            - "standard": idf = log(n_docs / df) + 1.0
            - "smooth": idf = log(n_docs + 1 / df + 1) + 1.0, i.e. 1 is added
              to all document frequencies, as if a single document containing
              every unique term was added to the corpus.
            - "bm25": idf = log((n_docs - df + 0.5) / (df + 0.5)), which is
              a form commonly used in information retrieval that allows for
              very common terms to receive negative weights.
            - None: no global weighting is applied to local term weights.

        dl_type: Type of document-length scaling to use for weights'
            normalization component:

            - "linear": dl (dls are already linear, so left as-is)
            - "sqrt": dl => sqrt(dl)
            - "log": dl => log(dl)
            - None: no normalization is applied to local (+global?) weights

        norm: If "l1" or "l2", normalize weights by the L1 or L2 norms, respectively,
            of row-wise vectors; otherwise, don't.
        min_df: Minimum number of documents in which a term must appear for it to be
            included in the vocabulary and as a column in a transformed doc-term matrix.
            If float, value is the fractional proportion of the total number of docs,
            which must be in [0.0, 1.0]; if int, value is the absolute number.
        max_df: Maximum number of documents in which a term may appear for it to be
            included in the vocabulary and as a column in a transformed doc-term matrix.
            If float, value is the fractional proportion of the total number of docs,
            which must be in [0.0, 1.0]; if int, value is the absolute number.
        max_n_terms: If specified, only include terms whose document frequency is within
            the top ``max_n_terms``.
        vocabulary_terms: Mapping of unique term string to unique term id, or
            an iterable of term strings that gets converted into such a mapping.
            Note that, if specified, vectorized output will include *only* these terms.
        vocabulary_grps: Mapping of unique group string to unique group id, or
            an iterable of group strings that gets converted into such a mapping.
            Note that, if specified, vectorized output will include *only* these groups.

    Attributes:
        vocabulary_terms (dict[str, int]): Mapping of unique term string to unique
            term id, either provided on instantiation or generated by calling
            :meth:`GroupVectorizer.fit()` on a collection of tokenized documents.
        vocabulary_grps (dict[str, int]): Mapping of unique group string to unique
            group id, either provided on instantiation or generated by calling
            :meth:`GroupVectorizer.fit()` on a collection of tokenized documents.

    See Also:
        :class:`Vectorizer`
    """

    def __init__(
        self,
        *,
        tf_type: Literal["linear", "sqrt", "log", "binary"] = "linear",
        idf_type: Optional[Literal["standard", "smooth", "bm25"]] = None,
        dl_type: Optional[Literal["linear", "sqrt", "log"]] = None,
        norm: Optional[Literal["l1", "l2"]] = None,
        min_df: int | float = 1,
        max_df: int | float = 1.0,
        max_n_terms: Optional[int] = None,
        vocabulary_terms: Optional[dict[str, int] | Iterable[str]] = None,
        vocabulary_grps: Optional[dict[str, int] | Iterable[str]] = None,
    ):
        super().__init__(
            tf_type=tf_type,
            idf_type=idf_type,
            dl_type=dl_type,
            norm=norm,
            min_df=min_df,
            max_df=max_df,
            max_n_terms=max_n_terms,
            vocabulary_terms=vocabulary_terms,
        )
        # now do the same thing for grps as was done for terms
        self.vocabulary_grps, self._fixed_grps = self._validate_vocabulary(
            vocabulary_grps
        )
        self.id_to_grp_: dict[int, str] = {}

    @property
    def id_to_grp(self) -> dict[int, str]:
        """
        Mapping of unique group id (int) to unique group string (str), i.e.
        the inverse of :attr:`GroupVectorizer.vocabulary_grps`. This attribute
        is only generated if needed, and it is automatically kept in sync
        with the corresponding vocabulary.
        """
        if len(self.id_to_grp_) != self.vocabulary_grps:
            self.id_to_grp_ = {
                grp_id: grp_str for grp_str, grp_id in self.vocabulary_grps.items()
            }
        return self.id_to_grp_

    # @id_to_grp.setter
    # def id_to_grp(self, new_id_to_grp):
    #     self.id_to_grp_ = new_id_to_grp
    #     self.vocabulary_grps = {
    #         grp_str: grp_id for grp_id, grp_str in new_id_to_grp.items()}

    @property
    def grps_list(self) -> list[str]:
        """
        List of group strings in row order of vectorized outputs. For example,
        ``grps_list[0]`` gives the group assigned to the first row in an
        output group-term-matrix, ``grp_term_matrix[0, :]``.
        """
        self._check_vocabulary()
        return [
            grp_str
            for grp_str, _ in sorted(
                self.vocabulary_grps.items(), key=operator.itemgetter(1)
            )
        ]

    def fit(
        self, tokenized_docs: Iterable[Iterable[str]], grps: Iterable[str]
    ) -> "GroupVectorizer":
        """
        Count terms in ``tokenized_docs`` and, if not already provided, build up
        a vocabulary based those terms; do the same for the groups in ``grps``.
        Fit and store global weights (IDFs) and, if needed for term weighting,
        the average document length.

        Args:
            tokenized_docs: A sequence of tokenized documents, where each is
                a sequence of term strings. For example::

                    >>> ([tok.lemma_ for tok in spacy_doc]
                    ...  for spacy_doc in spacy_docs)
                    >>> ((ne.text for ne in extract.entities(doc))
                    ...  for doc in corpus)

            grps: Sequence of group names by which the terms in ``tokenized_docs``
                are aggregated, where the first item in ``grps`` corresponds to
                the first item in ``tokenized_docs``, and so on.

        Returns:
            GroupVectorizer instance that has just been fit.
        """
        _ = self._fit(tokenized_docs, grps)
        return self

    def fit_transform(
        self, tokenized_docs: Iterable[Iterable[str]], grps: Iterable[str]
    ) -> sp.csr_matrix:
        """
        Count terms in ``tokenized_docs`` and, if not already provided, build up
        a vocabulary based those terms; do the same for the groups in ``grps``.
        Fit and store global weights (IDFs) and, if needed for term weighting,
        the average document length. Transform ``tokenized_docs`` into a
        group-term matrix with values weighted according to the parameters in
        :class:`GroupVectorizer` initialization.

        Args:
            tokenized_docs: A sequence of tokenized documents, where each is
                a sequence of term strings. For example::

                    >>> ([tok.lemma_ for tok in spacy_doc]
                    ...  for spacy_doc in spacy_docs)
                    >>> ((ne.text for ne in extract.entities(doc))
                    ...  for doc in corpus)

            grps: Sequence of group names by which the terms in ``tokenized_docs``
                are aggregated, where the first item in ``grps`` corresponds to
                the first item in ``tokenized_docs``, and so on.

        Returns:
            The transformed group-term matrix, where rows correspond to groups
            and columns correspond to terms, as a sparse row matrix.
        """
        # count terms and fit global weights
        grp_term_matrix = self._fit(tokenized_docs, grps)
        # re-weight values in group-term matrix, as specified in init
        grp_term_matrix = self._reweight_values(grp_term_matrix)
        return grp_term_matrix

    def transform(
        self, tokenized_docs: Iterable[Iterable[str]], grps: Iterable[str]
    ) -> sp.csr_matrix:
        """
        Transform ``tokenized_docs`` and ``grps`` into a group-term matrix with
        values weighted according to the parameters in :class:`GroupVectorizer`
        initialization and the global weights computed by calling
        :meth:`GroupVectorizer.fit()`.

        Args:
            tokenized_docs: A sequence of tokenized documents, where each is
                a sequence of term strings. For example::

                    >>> ([tok.lemma_ for tok in spacy_doc]
                    ...  for spacy_doc in spacy_docs)
                    >>> ((ne.text for ne in extract.entities(doc))
                    ...  for doc in corpus)

            grps: Sequence of group names by which the terms in ``tokenized_docs``
                are aggregated, where the first item in ``grps`` corresponds to
                the first item in ``tokenized_docs``, and so on.

        Returns:
            The transformed group-term matrix, where rows correspond to groups
            and columns correspond to terms, as a sparse row matrix.

        Note:
            For best results, the tokenization used to produce ``tokenized_docs``
            should be the same as was applied to the docs used in fitting this
            vectorizer or in generating a fixed input vocabulary.

            Consider an extreme case where the docs used in fitting consist of
            lowercased (non-numeric) terms, while the docs to be transformed are
            all uppercased: The output group-term-matrix will be empty.
        """
        self._check_vocabulary()
        grp_term_matrix, _, _ = self._count_terms(tokenized_docs, grps, True, True)
        return self._reweight_values(grp_term_matrix)

    def _fit(
        self, tokenized_docs: Iterable[Iterable[str]], grps: Iterable[str]
    ) -> sp.csr_matrix:
        """
        Count terms and, if :attr:`Vectorizer.fixed_terms` is False, build up
        a vocabulary based on the terms found in ``tokenized_docs``. Transform
        ``tokenized_docs`` into a document-term matrix with absolute tf weights.
        Store global weights (IDFs) and, if :attr:`Vectorizer.doc_length_norm`
        is not None, the average doc length.

        Args:
            tokenized_docs
            grps

        Returns:
            group-term matrix
        """
        # count terms and, if not provided on init, build up a vocabulary
        grp_term_matrix, vocabulary_terms, vocabulary_grps = self._count_terms(
            tokenized_docs, grps, self._fixed_terms, self._fixed_grps
        )

        if self._fixed_terms is False:
            # filter terms by doc freq or info content, as specified in init
            grp_term_matrix, vocabulary_terms = self._filter_terms(
                grp_term_matrix, vocabulary_terms
            )
            # sort features alphabetically (vocabulary_terms modified in-place)
            grp_term_matrix = self._sort_vocab_and_matrix(
                grp_term_matrix, vocabulary_terms, axis="columns"
            )
            # *now* vocabulary_terms are known and fixed
            self.vocabulary_terms = vocabulary_terms
            self._fixed_terms = True
        if self._fixed_grps is False:
            # sort groups alphabetically (vocabulary_grps modified in-place)
            grp_term_matrix = self._sort_vocab_and_matrix(
                grp_term_matrix, vocabulary_grps, axis="rows"
            )
            # *now* vocabulary_grps are known and fixed
            self.vocabulary_grps = vocabulary_grps
            self._fixed_grps = True

        n_grps, n_terms = grp_term_matrix.shape

        if self.idf_type:
            # store the global weights as a diagonal sparse matrix of idfs
            idfs = get_inverse_doc_freqs(grp_term_matrix, type_=self.idf_type)
            self._idf_diag = sp.spdiags(
                idfs, diags=0, m=n_terms, n=n_terms, format="csr"
            )

        if self.tf_type == "bm25" and self.dl_type:
            # store the avg document length, used in bm25 weighting to normalize
            # term weights by the length of the containing documents
            self._avg_doc_length = get_doc_lengths(
                grp_term_matrix, type_=self.dl_type
            ).mean()

        return grp_term_matrix

    def _count_terms(
        self,
        tokenized_docs: Iterable[Iterable[str]],
        grps: Iterable[str],
        fixed_vocab_terms: bool,
        fixed_vocab_grps: bool,
    ) -> tuple[sp.csr_matrix, dict[str, int], dict[str, int]]:
        """
        Count terms and build up a vocabulary based on the terms found in the
        ``tokenized_docs`` and the groups found in ``grps``.

        Args:
            tokenized_docs
            grps
            fixed_vocab_terms
            fixed_vocab_grps

        Returns:
            Group-term matrix, terms vocabulary, groups vocabulary.
        """
        # TODO: can we adapt the optimization from `Vectorizer._count_terms()` here?
        if fixed_vocab_terms is False:
            # add a new value when a new term is seen
            vocabulary_terms: Union[dict, DefaultDict] = collections.defaultdict()
            vocabulary_terms.default_factory = vocabulary_terms.__len__
        else:
            vocabulary_terms = self.vocabulary_terms

        if fixed_vocab_grps is False:
            # add a new value when a new group is seen
            vocabulary_grps = collections.defaultdict()
            vocabulary_grps.default_factory = vocabulary_grps.__len__
        else:
            vocabulary_grps = self.vocabulary_grps

        data = array(str("i"))
        cols = array(str("i"))
        rows = array(str("i"))
        for grp, terms in zip(grps, tokenized_docs):
            try:
                grp_idx = vocabulary_grps[grp]
            except KeyError:
                # ignore out-of-vocabulary groups when fixed_grps=True
                continue

            term_counter = collections.defaultdict(int)
            for term in terms:
                try:
                    term_idx = vocabulary_terms[term]
                    term_counter[term_idx] += 1
                except KeyError:
                    # ignore out-of-vocabulary terms when fixed_terms=True
                    continue

            data.extend(term_counter.values())
            cols.extend(term_counter.keys())
            rows.extend(grp_idx for _ in range(len(term_counter)))

        # do we still want defaultdict behaviour?
        if fixed_vocab_terms is False:
            vocabulary_terms = dict(vocabulary_terms)
        if fixed_vocab_grps is False:
            vocabulary_grps = dict(vocabulary_grps)

        data = np.frombuffer(data, dtype=np.intc)
        rows = np.frombuffer(rows, dtype=np.intc)
        cols = np.frombuffer(cols, dtype=np.intc)

        grp_term_matrix = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(len(vocabulary_grps), len(vocabulary_terms)),
            dtype=np.int32,
        )
        grp_term_matrix.sort_indices()

        return grp_term_matrix, vocabulary_terms, vocabulary_grps
