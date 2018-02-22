"""
Represent a collection of spacy-processed texts as a document-term matrix of shape
(# docs, # unique terms), with a variety of filtering, normalization, and term
weighting schemes for the values.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import itertools
import operator
from array import array

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize as normalize_mat

from . import compat


BM25_K1 = 1.6  # value typically bounded in [1.2, 2.0]
BM25_B = 0.75


class Vectorizer(object):
    """
    Transform one or more tokenized documents into a document-term matrix of
    shape (# docs, # unique terms), with tf-, tf-idf, or binary-weighted values.

    Stream a corpus with metadata from disk::

        >>> cw = textacy.datasets.CapitolWords()
        >>> text_stream, metadata_stream = textacy.io.split_records(
        ...     cw.records(limit=1000), 'text', itemwise=False)
        >>> corpus = textacy.Corpus('en', texts=text_stream, metadatas=metadata_stream)
        >>> corpus
        Corpus(1000 docs; 538172 tokens)

    Tokenize and vectorize the first 600 documents of this corpus::

        >>> tokenized_docs = (
        ...     doc.to_terms_list(ngrams=1, named_entities=True, as_strings=True)
        ...     for doc in corpus[:600])
        >>> vectorizer = Vectorizer(
        ...     weighting='tfidf', normalize=True, smooth_idf=True,
        ...     min_df=3, max_df=0.95)
        >>> doc_term_matrix = vectorizer.fit_transform(tokenized_docs)
        >>> doc_term_matrix
        <600x4346 sparse matrix of type '<class 'numpy.float64'>'
        	    with 69673 stored elements in Compressed Sparse Row format>

    Tokenize and vectorize the remaining 400 documents of the corpus, using only
    the groups, terms, and weights learned in the previous step:

        >>> tokenized_docs = (
        ...     doc.to_terms_list(ngrams=1, named_entities=True, as_strings=True)
        ...     for doc in corpus[600:])
        >>> doc_term_matrix = vectorizer.transform(tokenized_docs)
        >>> doc_term_matrix
        <400x4346 sparse matrix of type '<class 'numpy.float64'>'
        	    with 38756 stored elements in Compressed Sparse Row format>

    Inspect the terms associated with columns:

        >>> vectorizer.terms_list[:5]  # NOTE: that empty string shouldn't be there :/
        ['speaker', '', 'republican', 'house', 'american']

    If known in advance, limit the terms included in vectorized outputs
    to a particular set of values::

        >>> tokenized_docs = (
        ...     doc.to_terms_list(ngrams=1, named_entities=True, as_strings=True)
        ...     for doc in corpus[:600])
        >>> vectorizer = Vectorizer(
        ...     weighting='tfidf', normalize=True, smooth_idf=True,
        ...     min_df=3, max_df=0.95,
        ...     vocabulary_terms=['president', 'bill', 'unanimous', 'distinguished', 'american'])
        >>> doc_term_matrix = vectorizer.fit_transform(tokenized_docs)
        >>> doc_term_matrix
        <600x5 sparse matrix of type '<class 'numpy.float64'>'
	            with 844 stored elements in Compressed Sparse Row format>
        >>> vectorizer.terms_list
        ['american', 'bill', 'distinguished', 'president', 'unanimous']

    Args:
        weighting ({'tf', 'tfidf', 'bm25', 'binary'}): Overall weighting scheme
            used to assign values in a doc-term matrix. Note that certain aspects
            of these schemes may be modified or extended, depending on other args.

            - 'tf': Value (i, j) corresponds to the number of occurrences of
              term j in doc i, commonly referred to as its term frequency (tf).
              Terms appearing many times in a given doc receive a higher weight.
            - 'tfidf': Doc-specific, *local* term frequencies are multiplied by
              their corpus-wide, *global* inverse document frequencies (idfs).
              Terms appearing in many docs have a higher document frequency (df),
              a smaller idf, and thus a lower weight.
            - 'bm25': This scheme includes a local tf component that increases
              asymptotically, so higher tfs have diminishing effects on the overall
              weight; a global idf component that can go *negative* for terms
              that appear in a sufficiently high fraction of docs; as well as
              a row-wise normalization that accounts for doc length, such that
              terms in shorter docs hit the tf asymptote sooner than those in
              longer docs.
            - 'binary': All non-zero tfs are set equal to 1. That's it.

        tf_scale ({'sqrt', 'log'} or None): Scaling applied to term frequencies.

            - 'sqrt': tf => sqrt(tf)
            - 'log': tf => log(tf) + 1
            - None: term frequencies are left as-is

        idf_type ({'standard', 'smooth', 'bm25'}): Type of inverse document
            frequency (idf) formulation to use.

            - 'standard': idf = log(n_docs / df) + 1.0
            - 'smooth': idf = log(n_docs + 1 / df + 1) + 1.0, i.e. 1 is added
              to all document frequencies, as if a single document containing
              every unique term was added to the corpus.
            - 'bm25': idf = log((n_docs - df + 0.5) / (df + 0.5)), which is
              a form commonly used in information retrieval that allows for
              very common terms to receive negative weights.

        dl_norm (bool or None): If True, normalize weights by doc length,
            i.e. divide by the total number of in-vocabulary terms appearing
            in a given doc; if False, don't normalize weights by doc length.
            If None, this normalization is applied in accordance with the standard
            form of the weighting scheme specified in ``weighting``. In effect,
            it's only applied (by default) for 'bm25'.
        dl_scale ({'sqrt', 'log'} or None): Scaling applied to doc lengths.

            - 'sqrt': dl => sqrt(dl)
            - 'log': dl => log(dl)
            - None: doc lengths are left as-is

        norm ({'l1', 'l2'} or None): If 'l1' or 'l2', normalize weights by the
            L1 or L2 norms, respectively, of row-wise vectors; otherwise,
            don't normalize the weights.
        vocabulary_terms (Dict[str, int] or Iterable[str]): Mapping of unique term
            string to unique term id, or an iterable of term strings that gets
            converted into a suitable mapping. Note that, if specified, vectorized
            outputs will include *only* these terms as columns.
        min_df (float or int): If float, value is the fractional proportion of
            the total number of documents, which must be in [0.0, 1.0]. If int,
            value is the absolute number. Filter terms whose document frequency
            is less than ``min_df``.
        max_df (float or int): If float, value is the fractional proportion of
            the total number of documents, which must be in [0.0, 1.0]. If int,
            value is the absolute number. Filter terms whose document frequency
            is greater than ``max_df``.
        min_ic (float): Filter terms whose information content is less than
            ``min_ic``; value must be in [0.0, 1.0].
        max_n_terms (int): Only include terms whose document frequency is within
            the top ``max_n_terms``.

    Attributes:
        vocabulary_terms (Dict[str, int]): Mapping of unique term string to unique
            term id, either provided on instantiation or generated by calling
            :meth:`Vectorizer.fit()` on a collection of tokenized documents.
        id_to_term (Dict[int, str]): Mapping of unique term id to unique term
            string, i.e. the inverse of :attr:`Vectorizer.vocabulary_terms`.
            This mapping is only generated as needed.
        terms_list (List[str]): List of term strings in column order of
            vectorized outputs.
    """

    def __init__(self,
                 weighting='tf', tf_scale=None, idf_type='smooth',
                 dl_norm=None, dl_scale=None, norm=None,
                 min_df=1, max_df=1.0, min_ic=0.0, max_n_terms=None,
                 vocabulary_terms=None):
        # sanity check numeric arguments
        if min_df < 0 or max_df < 0:
            raise ValueError('`min_df` and `max_df` must be positive numbers or None')
        if min_ic < 0.0 or min_ic > 1.0:
            raise ValueError('`min_ic` must be a float in the interval [0.0, 1.0]')
        if max_n_terms and max_n_terms < 0:
            raise ValueError('`max_n_terms` must be a positive integer or None')
        self.weighting = weighting
        self.tf_scale = tf_scale
        self.idf_type = idf_type
        self.dl_norm = dl_norm
        self.dl_scale = dl_scale
        self.norm = norm
        self.min_df = min_df
        self.max_df = max_df
        self.min_ic = min_ic
        self.max_n_terms = max_n_terms
        self.vocabulary_terms, self._fixed_terms = self._validate_vocabulary(vocabulary_terms)
        self.id_to_term_ = {}
        self._idf_diag = None
        self._avg_doc_length = None

    def _validate_vocabulary(self, vocabulary):
        """
        Validate an input vocabulary. If it's a mapping, ensure that term ids
        are unique and compact (i.e. without any gaps between 0 and the number
        of terms in ``vocabulary``. If it's a sequence, sort terms then assign
        integer ids in ascending order.
        """
        if vocabulary is not None:
            if not isinstance(vocabulary, collections.Mapping):
                vocab = {}
                for i, term in enumerate(sorted(vocabulary)):
                    if vocab.setdefault(term, i) != i:
                        raise ValueError(
                            'Terms in `vocabulary` must be unique, but "{}" '
                            'was found more than once.'.format(term))
                vocabulary = vocab
            else:
                ids = set(vocabulary.values())
                if len(ids) != len(vocabulary):
                    counts = collections.Counter(vocabulary.values())
                    n_dupe_term_ids = sum(
                        1 for term_id, term_id_count in counts.items()
                        if term_id_count > 1)
                    raise ValueError(
                        'Term ids in `vocabulary` must be unique, but {} ids'
                        'were assigned to more than one term.'.format(n_dupe_term_ids))
                for i in compat.range_(len(vocabulary)):
                    if i not in ids:
                        raise ValueError(
                            'Term ids in `vocabulary` must be compact, i.e. '
                            'not have any gaps, but term id {} is missing from '
                            'a vocabulary of {} terms'.format(i, len(vocabulary)))
            if not vocabulary:
                raise ValueError('`vocabulary` must not be empty.')
            is_fixed = True
        else:
            is_fixed = False
        return vocabulary, is_fixed

    def _check_vocabulary(self):
        """
        Check that instance has a valid vocabulary mapping;
        if not, raise a ValueError.
        """
        if not isinstance(self.vocabulary_terms, collections.Mapping):
            raise ValueError(
                'vocabulary hasn\'t been built; call `Vectorizer.fit()`')
        if len(self.vocabulary_terms) == 0:
            raise ValueError('vocabulary is empty')

    @property
    def id_to_term(self):
        """
        dict: Mapping of unique term id (int) to unique term string (str), i.e.
            the inverse of :attr:`Vectorizer.vocabulary`. This attribute is only
            generated if needed, and it is automatically kept in sync with the
            corresponding vocabulary.
        """
        if len(self.id_to_term_) != self.vocabulary_terms:
            self.id_to_term_ = {
                term_id: term_str for term_str, term_id in self.vocabulary_terms.items()}
        return self.id_to_term_

    # TODO: Do we *want* to allow setting to this property?
    # @id_to_term.setter
    # def id_to_term(self, new_id_to_term):
    #     self.id_to_term_ = new_id_to_term
    #     self.vocabulary_terms = {
    #         term_str: term_id for term_id, term_str in new_id_to_term.items()}

    @property
    def terms_list(self):
        """
        List of term strings in column order of vectorized outputs. For example,
        ``terms_list[0]`` gives the term assigned to the first column in an
        output doc-term-matrix, ``doc_term_matrix[:, 0]``.
        """
        self._check_vocabulary()
        return [term_str for term_str, _
                in sorted(self.vocabulary_terms.items(), key=operator.itemgetter(1))]

    def fit(self, tokenized_docs):
        """
        Count terms in ``tokenized_docs`` and, if not already provided, build up
        a vocabulary based those terms. Fit and store global weights (IDFs)
        and, if needed for term weighting, the average document length.

        Args:
            tokenized_docs (Iterable[Iterable[str]]): A sequence of tokenized
                documents, where each is a sequence of (str) terms. For example::

                    >>> ([tok.lemma_ for tok in spacy_doc]
                    ...  for spacy_doc in spacy_docs)
                    >>> ((ne.text for ne in extract.named_entities(doc))
                    ...  for doc in corpus)
                    >>> (doc.to_terms_list(as_strings=True)
                    ...  for doc in docs)

        Returns:
            :class:`Vectorizer`: The instance that has just been fit.
        """
        _ = self._fit(tokenized_docs)
        return self

    def fit_transform(self, tokenized_docs):
        """
        Count terms in ``tokenized_docs`` and, if not already provided, build up
        a vocabulary based those terms. Fit and store global weights (IDFs)
        and, if needed for term weighting, the average document length.
        Transform ``tokenized_docs`` into a document-term matrix with values
        weighted according to the parameters in :class:`Vectorizer` initialization.

        Args:
            tokenized_docs (Iterable[Iterable[str]]): A sequence of tokenized
                documents, where each is a sequence of (str) terms. For example::

                    >>> ([tok.lemma_ for tok in spacy_doc]
                    ...  for spacy_doc in spacy_docs)
                    >>> ((ne.text for ne in extract.named_entities(doc))
                    ...  for doc in corpus)
                    >>> (doc.to_terms_list(as_strings=True)
                    ...  for doc in docs)

        Returns:
            :class:`scipy.sparse.csr_matrix`: The transformed document-term matrix.
            Rows correspond to documents and columns correspond to terms.
        """
        # count terms and fit global weights
        doc_term_matrix = self._fit(tokenized_docs)
        # re-weight values in doc-term matrix, as specified in init
        doc_term_matrix = self._reweight_values(doc_term_matrix)
        return doc_term_matrix

    def transform(self, tokenized_docs):
        """
        Transform ``tokenized_docs`` into a document-term matrix with values
        weighted according to the parameters in :class:`Vectorizer` initialization
        and the global weights computed by calling :meth:`Vectorizer.fit()`.

        Args:
            tokenized_docs (Iterable[Iterable[str]]): A sequence of tokenized
                documents, where each is a sequence of (str) terms. For example::

                    >>> ([tok.lemma_ for tok in spacy_doc]
                    ...  for spacy_doc in spacy_docs)
                    >>> ((ne.text for ne in extract.named_entities(doc))
                    ...  for doc in corpus)
                    >>> (doc.to_terms_list(as_strings=True)
                    ...  for doc in docs)

        Returns:
            :class:`scipy.sparse.csr_matrix`: The transformed document-term matrix.
            Rows correspond to documents and columns correspond to terms.

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

    def _fit(self, tokenized_docs):
        """
        Count terms and, if :attr:`Vectorizer.fixed_terms` is False, build up
        a vocabulary based on the terms found in ``tokenized_docs``. Transform
        ``tokenized_docs`` into a document-term matrix with absolute tf weights.
        Store global weights (IDFs) and, if :attr:`Vectorizer.doc_length_norm`
        is not None, the average doc length.

        Args:
            tokenized_docs (Iterable[Iterable[str]]): A sequence of tokenized
                documents, where each is a sequence of (str) terms.

        Returns:
            :class:`scipy.sparse.csr_matrix`
        """
        # count terms and, if not provided on init, build up a vocabulary
        doc_term_matrix, vocabulary_terms = self._count_terms(
            tokenized_docs, self._fixed_terms)

        if self._fixed_terms is False:
            # filter terms by doc freq or info content, as specified in init
            doc_term_matrix, vocabulary_terms = self._filter_terms(
                doc_term_matrix, vocabulary_terms)
            # sort features alphabetically (vocabulary_terms modified in-place)
            doc_term_matrix = self._sort_terms(
                doc_term_matrix, vocabulary_terms)
            # *now* vocabulary_terms are known and fixed
            self.vocabulary_terms = vocabulary_terms
            self._fixed_terms = True

        n_docs, n_terms = doc_term_matrix.shape

        if self.weighting in ('tfidf', 'bm25'):
            # store the global weights as a diagonal sparse matrix of idfs
            idfs = get_inverse_doc_freqs(doc_term_matrix, type_=self.idf_type)
            self._idf_diag = sp.spdiags(
                idfs, diags=0, m=n_terms, n=n_terms, format='csr')

        if self.weighting == 'bm25' and self.dl_norm is not False:
            # store the avg document length, used in bm25 weighting to normalize
            # term weights by the length of the containing documents
            self._avg_doc_length = get_doc_lengths(
                doc_term_matrix, scale=self.dl_scale).mean()

        return doc_term_matrix

    def _count_terms(self, tokenized_docs, fixed_vocab):
        """
        Count terms found in ``tokenized_docs`` and, if ``fixed_vocab`` is False,
        build up a vocabulary based on those terms.

        Args:
            tokenized_docs (Iterable[Iterable[str]]): A sequence of tokenized
                documents, where each is a sequence of (str) terms. For example::

                    >>> ([tok.lemma_ for tok in spacy_doc]
                    ...  for spacy_doc in spacy_docs)
                    >>> ((ne.text for ne in extract.named_entities(doc))
                    ...  for doc in corpus)
                    >>> (doc.to_terms_list(as_strings=True)
                    ...  for doc in docs)

            fixed_vocab (bool): If False, a new vocabulary is built from terms
                in ``tokenized_docs``; if True, only terms already found in
                :attr:`Vectorizer.vocabulary_terms` are counted.

        Returns:
            :class:`scipy.sparse.csr_matrix`, Dict[str, int]
        """
        if fixed_vocab is False:
            # add a new value when a new term is seen
            vocabulary = collections.defaultdict()
            vocabulary.default_factory = vocabulary.__len__
        else:
            vocabulary = self.vocabulary_terms

        indices = array(str('i'))
        indptr = array(str('i'), [0])
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
            dtype=np.int32)
        doc_term_matrix.sum_duplicates()

        # pretty sure this is a good thing to do... o_O
        doc_term_matrix.sort_indices()

        return doc_term_matrix, vocabulary

    def _filter_terms(self, doc_term_matrix, vocabulary):
        """
        Filter terms in ``vocabulary`` by their document frequency or information
        content, as specified in :class:`Vectorizer` initialization.

        Args:
            doc_term_matrix (:class:`sp.sparse.csr_matrix`): Sparse matrix of
                shape (# docs, # unique terms), where value (i, j) is the weight
                of term j in doc i.
            vocabulary (Dict[str, int]): Mapping of term strings to their unique
                integer ids, e.g. ``{"hello": 0, "world": 1}``.

        Returns:
            :class:`scipy.sparse.csr_matrix`, Dict[str, int]
        """
        if self.max_df != 1.0 or self.min_df != 1 or self.max_n_terms is not None:
            doc_term_matrix, vocabulary = filter_terms_by_df(
                doc_term_matrix, vocabulary,
                max_df=self.max_df, min_df=self.min_df, max_n_terms=self.max_n_terms)
        if self.min_ic != 0.0:
            doc_term_matrix, vocabulary = filter_terms_by_ic(
                doc_term_matrix, vocabulary,
                min_ic=self.min_ic, max_n_terms=self.max_n_terms)
        return doc_term_matrix, vocabulary

    def _sort_terms(self, doc_term_matrix, vocabulary):
        """
        Sort terms in ``vocabulary`` alphabetically, modifying the vocabulary
        in-place, and returning a correspondingly reordered ``doc_term_matrix``.

        Args:
            doc_term_matrix (:class:`sp.sparse.csr_matrix`)
            vocabulary (Dict[str, int])

        Returns:
            :class:`scipy.sparse.csr_matrix`
        """
        sorted_terms = sorted(vocabulary.items())
        new_idx_array = np.empty(len(sorted_terms), dtype=np.int32)
        for new_idx, (term, old_idx) in enumerate(sorted_terms):
            new_idx_array[new_idx] = old_idx
            vocabulary[term] = new_idx
        # use fancy indexing to reorder columns
        return doc_term_matrix[:, new_idx_array]

    def _reweight_values(self, doc_term_matrix):
        """
        Re-weight values in a doc-term matrix according to parameters specified
        in :class:`Vectorizer` initialization: binary or tf-idf weighting,
        sublinear term-frequency, document-normalized weights.

        Args:
            doc_term_matrix (:class:`sp.sparse.csr_matrix`): Sparse matrix of
                shape (# docs, # unique terms), where value (i, j) is the weight
                of term j in doc i.

        Returns:
            :class:`scipy.sparse.csr_matrix`: Re-weighted doc-term matrix.
        """
        # re-weight the local components (term freqs)
        if self.weighting == 'binary':
            doc_term_matrix.data.fill(1)
        elif self.weighting == 'bm25':
            if self.dl_norm is False:
                doc_term_matrix.data = (
                    doc_term_matrix.data * (BM25_K1 + 1.0) /
                    (BM25_K1 + doc_term_matrix.data)
                )
            else:
                dls = vsm.get_doc_lengths(doc_term_matrix, scale=self.dl_scale)
                length_norm = (1 - BM25_B) + (BM25_B * (dls / self.avg_doc_length))
                doc_term_matrix = doc_term_matrix.tocoo(copy=False)
                doc_term_matrix.data = (
                    doc_term_matrix.data * (BM25_K1 + 1.0) /
                    (doc_term_matrix.data + (BM25_K1 * length_norm[doc_term_matrix.row]))
                )
                doc_term_matrix = doc_term_matrix.tocsr(copy=False)
        else:  # tf or tfidf
            if self.tf_scale == 'sqrt':
                _ = np.sqrt(doc_term_matrix.data, doc_term_matrix.data)
            elif self.tf_scale == 'log':
                _ = np.log(doc_term_matrix.data, doc_term_matrix.data)
                doc_term_matrix.data += 1.0

        # apply the global component (idfs), column-wise
        if self.weighting in ('tfidf', 'bm25'):
            doc_term_matrix = doc_term_matrix * self._idf_diag

        # apply normalizations, row-wise
        if self.weighting != 'bm25' and self.dl_norm is True:
            # we've already handled doc-length normalization in bm25
            dls = get_doc_lengths(doc_term_matrix, scale=self.dl_scale)
            dl_diag = sp.spdiags(1.0 / dls, diags=0, m=n_docs, n=n_docs, format='csr')
            doc_term_matrix = dl_diag * doc_term_matrix
        if self.norm is not None:
            doc_term_matrix = normalize_mat(
                doc_term_matrix, norm=self.norm, axis=1, copy=False)

        return doc_term_matrix


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

        >>> cw = textacy.datasets.CapitolWords()
        >>> text_stream, metadata_stream = textacy.io.split_records(
        ...     cw.records(limit=1000), 'text', itemwise=False)
        >>> corpus = textacy.Corpus('en', texts=text_stream, metadatas=metadata_stream)
        >>> corpus
        Corpus(1000 docs; 538172 tokens)

    Tokenize and vectorize the first 600 documents of this corpus, where terms
    are grouped not by documents but by a categorical value in the docs' metadata::

        >>> tokenized_docs, groups = textacy.io.unzip(
        ...     (doc.to_terms_list(ngrams=1, named_entities=True, as_strings=True),
        ...      doc.metadata['speaker_name'])
        ...     for doc in corpus[:600])
        >>> vectorizer = GroupVectorizer(
        ...     weighting='tfidf', normalize=True, smooth_idf=True,
        ...     min_df=3, max_df=0.95)
        >>> grp_term_matrix = vectorizer.fit_transform(tokenized_docs, groups)
        >>> grp_term_matrix
        <5x1793 sparse matrix of type '<class 'numpy.float64'>'
        	    with 6075 stored elements in Compressed Sparse Row format>

    Tokenize and vectorize the remaining 400 documents of the corpus, using only
    the groups, terms, and weights learned in the previous step:

        >>> tokenized_docs, groups = textacy.io.unzip(
        ...     (doc.to_terms_list(ngrams=1, named_entities=True, as_strings=True),
        ...      doc.metadata['speaker_name'])
        ...     for doc in corpus[600:])
        >>> grp_term_matrix = vectorizer.transform(tokenized_docs, groups)
        >>> grp_term_matrix
        <5x1793 sparse matrix of type '<class 'numpy.float64'>'
        	    with 4440 stored elements in Compressed Sparse Row format>

    Inspect the terms associated with columns and groups associated with rows:

        >>> vectorizer.terms_list[:5]
        ['georgia', 'dole', 'virtually', 'worker', 'financial']
        >>> vectorizer.grps_list
        ['Bernie Sanders', 'Lindsey Graham', 'Rick Santorum', 'Joseph Biden', 'John Kasich']

    If known in advance, limit the terms and/or groups included in vectorized outputs
    to a particular set of values::

        >>> tokenized_docs, groups = textacy.io.unzip(
        ...     (doc.to_terms_list(ngrams=1, named_entities=True, as_strings=True),
        ...      doc.metadata['speaker_name'])
        ...     for doc in corpus[:600])
        >>> vectorizer = GroupVectorizer(
        ...     weighting='tfidf', normalize=True, smooth_idf=True,
        ...     min_df=3, max_df=0.95,
        ...     vocabulary_terms=['legislation', 'federal government', 'house', 'constitutional'],
        ...     vocabulary_grps=['Bernie Sanders', 'Lindsey Graham', 'Rick Santorum'])
        >>> grp_term_matrix = vectorizer.fit_transform(tokenized_docs, groups)
        >>> grp_term_matrix
        <3x4 sparse matrix of type '<class 'numpy.float64'>'
	            with 12 stored elements in Compressed Sparse Row format>
        >>> vectorizer.terms_list
        ['constitutional', 'federal government', 'house', 'legislation']
        >>> vectorizer.grps_list
        ['Bernie Sanders', 'Lindsey Graham', 'Rick Santorum']

    Args:
        weighting ({'tf', 'tfidf', 'binary'}): Weighting to assign to terms in
            the doc-term matrix. If 'tf', matrix values (i, j) correspond to the
            number of occurrences of term j in doc i; if 'tfidf', term frequencies
            (tf) are multiplied by their corresponding inverse document frequencies
            (idf); if 'binary', all non-zero values are set equal to 1.
        normalize (bool): If True, normalize term frequencies by the
            L2 norms of the vectors.
        sublinear_tf (bool): If True, apply sub-linear term-frequency scaling,
            i.e. tf => 1 + log(tf).
        smooth_idf (bool): If True, add 1 to all document frequencies, equivalent
            to adding a single document to the corpus containing every unique term.
        vocabulary_terms (Dict[str, int] or Iterable[str]): Mapping of unique term
            string to unique term id, or an iterable of term strings that gets
            converted into a suitable mapping. Note that, if specified, vectorized
            outputs will include *only* these terms as columns.
        vocabulary_grps (Dict[str, int] or Iterable[str]): Mapping of unique group
            string to unique group id, or an iterable of group strings that gets
            converted into a suitable mapping. Note that, if specified, vectorized
            outputs will include *only* these groups as rows.
        min_df (float or int): If float, value is the fractional proportion of
            the total number of documents (groups), which must be in [0.0, 1.0].
            If int, value is the absolute number. Filter terms whose document (group)
            frequency is less than ``min_df``.
        max_df (float or int): If float, value is the fractional proportion of
            the total number of documents (groups), which must be in [0.0, 1.0].
            If int, value is the absolute number. Filter terms whose document (group)
            frequency is greater than ``max_df``.
        min_ic (float): Filter terms whose information content is less than
            ``min_ic``; value must be in [0.0, 1.0].
        max_n_terms (int): Only include terms whose document (group) frequency
            is within the top ``max_n_terms``.

    Attributes:
        vocabulary_terms (Dict[str, int]): Mapping of unique term string to unique
            term id, either provided on instantiation or generated by calling
            :meth:`GroupVectorizer.fit()` on a collection of tokenized documents.
        vocabulary_grps (Dict[str, int]): Mapping of unique group string to unique
            group id, either provided on instantiation or generated by calling
            :meth:`GroupVectorizer.fit()` on a collection of tokenized documents.
        id_to_term (Dict[int, str]): Mapping of unique term id to unique term
            string, i.e. the inverse of :attr:`GroupVectorizer.vocabulary_terms`.
            This mapping is only generated as needed.
        id_to_grp (Dict[int, str]): Mapping of unique group id to unique group
            string, i.e. the inverse of :attr:`GroupVectorizer.vocabulary_grps`.
            This mapping is only generated as needed.
        terms_list (List[str]): List of term strings in column order of
            vectorized outputs.
        grps_list (List[str]): List of group strings in row order of
            vectorized outputs.

    See Also:
        :class:`Vectorizer`
    """

    def __init__(self,
                 weighting='tf', tf_scale=None, idf_type='smooth',
                 dl_norm=None, dl_scale=None, norm=None,
                 min_df=1, max_df=1.0, min_ic=0.0, max_n_terms=None,
                 vocabulary_terms=None, vocabulary_grps=None):
        super(GroupVectorizer, self).__init__(
            weighting=weighting, tf_scale=tf_scale, idf_type=idf_type,
            dl_norm=dl_norm, dl_scale=dl_scale, norm=norm,
            min_df=min_df, max_df=max_df, min_ic=min_ic, max_n_terms=max_n_terms,
            vocabulary_terms=vocabulary_terms)
        # now do the same thing for grps as was done for terms
        self.vocabulary_grps, self._fixed_grps = self._validate_vocabulary(vocabulary_grps)
        self.id_to_grp_ = {}

    @property
    def id_to_grp(self):
        """
        dict: Mapping of unique group id (int) to unique group string (str), i.e.
            the inverse of :attr:`GroupVectorizer.vocabulary_grps`. This attribute
            is only generated if needed, and it is automatically kept in sync
            with the corresponding vocabulary.
        """
        if len(self.id_to_grp_) != self.vocabulary_grps:
            self.id_to_grp_ = {
                grp_id: grp_str for grp_str, grp_id in self.vocabulary_grps.items()}
        return self.id_to_grp_

    # @id_to_grp.setter
    # def id_to_grp(self, new_id_to_grp):
    #     self.id_to_grp_ = new_id_to_grp
    #     self.vocabulary_grps = {
    #         grp_str: grp_id for grp_id, grp_str in new_id_to_grp.items()}

    @property
    def grps_list(self):
        """
        List of group strings in row order of vectorized outputs. For example,
        ``grps_list[0]`` gives the group assigned to the first row in an
        output group-term-matrix, ``grp_term_matrix[0, :]``.
        """
        self._check_vocabulary()
        return [grp_str for grp_str, _
                in sorted(self.vocabulary_grps.items(), key=operator.itemgetter(1))]

    def fit(self, tokenized_docs, grps):
        """
        Count terms and build up a vocabulary based on the terms found in the
        input ``tokenized_docs``. Also build a vocabulary for the groups in ``grps``.

        Args:
            tokenized_docs (Iterable[Iterable[str]]): A sequence of tokenized
                documents, where each is a sequence of (str) terms. For example::

                    >>> ([tok.lemma_ for tok in spacy_doc]
                    ...  for spacy_doc in spacy_docs)
                    >>> ((ne.text for ne in extract.named_entities(doc))
                    ...  for doc in corpus)
                    >>> (doc.to_terms_list(as_strings=True)
                    ...  for doc in docs)

            grps (Iterable[str]): Sequence of group names by which the terms in
                ``tokenized_docs`` are aggregated, where the first item in ``grps``
                corresponds to the first item in ``tokenized_docs``, and so on.

        Returns:
            :class:`GroupVectorizer`: The instance that has just been fit.
        """
        _ = self.fit_transform(tokenized_docs, grps)
        return self

    def fit_transform(self, tokenized_docs, grps):
        """
        Count terms and build up a vocabulary based on the terms found in the
        input ``tokenized_docs``. Also build a vocabulary for the groups in ``grps``.
        Then transform ``tokenized_docs`` into a group-term matrix with values
        weighted according to the parameters specified in
        :class:`GroupVectorizer` initialization.

        Args:
            tokenized_docs (Iterable[Iterable[str]]): A sequence of tokenized
                documents, where each is a sequence of (str) terms. For example::

                    >>> ([tok.lemma_ for tok in spacy_doc]
                    ...  for spacy_doc in spacy_docs)
                    >>> ((ne.text for ne in extract.named_entities(doc))
                    ...  for doc in corpus)
                    >>> (doc.to_terms_list(as_strings=True)
                    ...  for doc in docs)

            grps (Iterable[str]): Sequence of group names by which the terms in
                ``tokenized_docs`` are aggregated, where the first item in ``grps``
                corresponds to the first item in ``tokenized_docs``, and so on.

        Returns:
            :class:`scipy.sparse.csr_matrix`: The transformed group-term matrix.
            Rows correspond to groups and columns correspond to terms.
        """
        # count terms and build up a vocabulary
        grp_term_matrix, self.vocabulary_terms, self.vocabulary_grps = self._count_terms(
            tokenized_docs, grps, self._fixed_terms, self._fixed_grps)
        # filter terms by group freq or info content, as specified in init
        grp_term_matrix, self.vocabulary_terms = self._filter_terms(
            grp_term_matrix, self.vocabulary_terms)
        # re-weight values in group-term matrix, as specified in init
        grp_term_matrix = self._reweight_values(grp_term_matrix)
        return grp_term_matrix

    def transform(self, tokenized_docs, grps):
        """
        Transform ``tokenized_docs`` into a group-term matrix, with columns
        determined from calling :meth:`GroupVectorizer.fit()` or from providing
        a fixed vocabulary when initializing :class:`GroupVectorizer`, and values
        weighted according to the parameters specified in class initialization.

        Args:
            tokenized_docs (Iterable[Iterable[str]]): A sequence of tokenized
                documents, where each is a sequence of (str) terms. For example::

                    >>> ([tok.lemma_ for tok in spacy_doc]
                    ...  for spacy_doc in spacy_docs)
                    >>> ((ne.text for ne in extract.named_entities(doc))
                    ...  for doc in corpus)
                    >>> (doc.to_terms_list(as_strings=True)
                    ...  for doc in docs)

            grps (Iterable[str]): Sequence of group names by which the terms in
                ``tokenized_docs`` are aggregated, where the first item in ``grps``
                corresponds to the first item in ``tokenized_docs``, and so on.

        Returns:
            :class:`scipy.sparse.csr_matrix`: The transformed group-term matrix.
            Rows correspond to groups and columns correspond to terms.

        Note:
            For best results, the tokenization used to produce ``tokenized_docs``
            should be the same as was applied to the docs used in fitting this
            vectorizer or in generating a fixed input vocabulary.

            Consider an extreme case where the docs used in fitting consist of
            lowercased (non-numeric) terms, while the docs to be transformed are
            all uppercased: The output group-term-matrix will be empty.
        """
        self._check_vocabulary()
        grp_term_matrix, _, _ = self._count_terms(
            tokenized_docs, grps, True, True)
        return self._reweight_values(grp_term_matrix)

    def _fit(self, tokenized_docs, grps):
        """
        Count terms and, if :attr:`Vectorizer.fixed_terms` is False, build up
        a vocabulary based on the terms found in ``tokenized_docs``. Transform
        ``tokenized_docs`` into a document-term matrix with absolute tf weights.
        Store global weights (IDFs) and, if :attr:`Vectorizer.doc_length_norm`
        is not None, the average doc length.

        Args:
            tokenized_docs (Iterable[Iterable[str]]): A sequence of tokenized
                documents, where each is a sequence of (str) terms.

        Returns:
            :class:`scipy.sparse.csr_matrix`
        """
        # count terms and, if not provided on init, build up a vocabulary
        grp_term_matrix, vocabulary_terms, vocabulary_grps = self._count_terms(
            tokenized_docs, grps, self._fixed_terms, self._fixed_grps)

        if self._fixed_terms is False:
            # filter terms by doc freq or info content, as specified in init
            doc_term_matrix, vocabulary_terms = self._filter_terms(
                doc_term_matrix, vocabulary_terms)
            # sort features alphabetically (vocabulary_terms modified in-place)
            doc_term_matrix = self._sort_terms(
                doc_term_matrix, vocabulary_terms)
            # *now* vocabulary_terms are known and fixed
            self.vocabulary_terms = vocabulary_terms
            self._fixed_terms = True
        if self._fixed_grps is False:
            # sort groups alphabetically (vocabulary_grps modified in-place)
            grp_term_matrix = self._sort_grps(
                grp_term_matrix, vocabulary_grps)
            # *now* vocabulary_terms are known and fixed
            self.vocabulary_grps = vocabulary_grps
            self._fixed_grps = True

        n_grps, n_terms = doc_term_matrix.shape

        if self.weighting in ('tfidf', 'bm25'):
            # store the global weights as a diagonal sparse matrix of idfs
            idfs = get_inverse_doc_freqs(grp_term_matrix, type_=self.idf_type)
            self._idf_diag = sp.spdiags(
                idfs, diags=0, m=n_terms, n=n_terms, format='csr')

        if self.weighting == 'bm25' and self.dl_norm is not False:
            # store the avg document length, used in bm25 weighting to normalize
            # term weights by the length of the containing documents
            self._avg_doc_length = get_doc_lengths(
                grp_term_matrix, scale=self.dl_scale).mean()

        return grp_term_matrix

    def _count_terms(self, tokenized_docs, grps, fixed_vocab_terms, fixed_vocab_grps):
        """
        Count terms and build up a vocabulary based on the terms found in the
        ``tokenized_docs`` and the groups found in ``grps``.

        Args:
            tokenized_docs (Iterable[Iterable[str]]): A sequence of tokenized
                documents, where each is a sequence of (str) terms. For example::

                    >>> ([tok.lemma_ for tok in spacy_doc]
                    ...  for spacy_doc in spacy_docs)
                    >>> ((ne.text for ne in extract.named_entities(doc))
                    ...  for doc in corpus)
                    >>> (doc.to_terms_list(as_strings=True)
                    ...  for doc in docs)

            grps (Iterable[str]): Sequence of group names by which the terms in
                ``tokenized_docs`` are aggregated, where the first item in ``grps``
                corresponds to the first item in ``tokenized_docs``, and so on.
            fixed_vocab_terms (bool): If False, a new vocabulary is built from terms
                in ``tokenized_docs``; if True, only terms already found in the
                :attr:`GroupVectorizer.vocabulary_terms` are counted.
            fixed_vocab_grps (bool): If False, a new vocabulary is built from groups
                in ``grps``; if True, only groups already found in the
                :attr:`GroupVectorizer.vocabulary_grps` are counted.

        Returns:
            :class:`scipy.sparse.csr_matrix`, Dict[str, int], Dict[str, int]

        TODO: can we adapt the optimization from Vectorizer into this case?
        """
        if fixed_vocab_terms is False:
            # add a new value when a new term is seen
            vocabulary_terms = collections.defaultdict()
            vocabulary_terms.default_factory = vocabulary_terms.__len__
        else:
            vocabulary_terms = self.vocabulary_terms

        if fixed_vocab_grps is False:
            # add a new value when a new group is seen
            vocabulary_grps = collections.defaultdict()
            vocabulary_grps.default_factory = vocabulary_grps.__len__
        else:
            vocabulary_grps = self.vocabulary_grps

        data = array(str('i'))
        cols = array(str('i'))
        rows = array(str('i'))
        for grp, terms in compat.zip_(grps, tokenized_docs):

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
            rows.extend(grp_idx for _ in compat.range_(len(term_counter)))

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
            dtype=np.int32)
        grp_term_matrix.sort_indices()

        return grp_term_matrix, vocabulary_terms, vocabulary_grps

    def _sort_grps(self, grp_term_matrix, vocabulary):
        """
        Sort terms in ``vocabulary`` alphabetically, modifying the vocabulary
        in-place, and returning a correspondingly reordered ``grp_term_matrix``.

        Args:
            grp_term_matrix (:class:`sp.sparse.csr_matrix`)
            vocabulary (Dict[str, int])

        Returns:
            :class:`scipy.sparse.csr_matrix`

        TODO: Consolidate this with existing :attr:`Vectorizer._sort_terms`,
        just add an ``axis`` arg or something with 0/1 values.
        """
        sorted_terms = sorted(vocabulary.items())
        new_idx_array = np.empty(len(sorted_terms), dtype=np.int32)
        for new_idx, (term, old_idx) in enumerate(sorted_terms):
            new_idx_array[new_idx] = old_idx
            vocabulary[term] = new_idx
        # use fancy indexing to reorder columns
        return grp_term_matrix[new_idx_array, :]


def apply_idf_weighting(doc_term_matrix, smooth_idf=True):
    """
    Apply inverse document frequency (idf) weighting to a term-frequency (tf)
    weighted document-term matrix, optionally smoothing idf values.

    Args:
        doc_term_matrix (:class:`scipy.sparse.csr_matrix <scipy.sparse.csr_matrix`):
            M X N matrix, where M is the # of docs and N is the # of unique terms
        smooth_idf (bool): if True, add 1 to all document frequencies, equivalent
            to adding a single document to the corpus containing every unique term

    Returns:
        :class:`scipy.sparse.csr_matrix <scipy.sparse.csr_matrix>`: sparse matrix
        of shape (# docs, # unique terms), where value (i, j) is the tfidf
        weight of term j in doc i
    """
    idfs = get_inverse_doc_freqs(doc_term_matrix, smooth=smooth_idf)
    return doc_term_matrix.dot(sp.diags(idfs, 0))


def get_term_freqs(doc_term_matrix, scale=None, normalize=True):
    """
    Compute absolute or relative term frequencies for all terms in a
    document-term matrix, with optional sub-linear scaling.

    Args:
        doc_term_matrix (:class:`scipy.sparse.csr_matrix`): M X N sparse matrix,
            where M is the # of docs, N is the # of unique terms. Values are
            the *absolute* counts of term n per doc m.
        scale ({'sqrt', 'log'} or None): Scaling applied to absolute term counts.
            If 'sqrt', tf => sqrt(tf); if 'log', tf => log(tf) + 1.
            If None, term counts are returned as-is.
        normalize (bool): If True, return normalized term frequencies, i.e.
            term counts divided by the total number of terms; otherwise, return
            absolute term counts.

    Returns:
        :class:`numpy.ndarray <numpy.ndarray>`: Array of absolute or relative term
        frequencies, with length equal to the # of unique terms (# of columns)
        in ``doc_term_matrix``.

    Raises:
        ValueError: if ``doc_term_matrix`` doesn't have any non-zero entries
    """
    if doc_term_matrix.nnz == 0:
        raise ValueError('`doc_term_matrix` must have at least 1 non-zero entry')
    _, n_terms = doc_term_matrix.shape
    tfs = np.asarray(doc_term_matrix.sum(axis=0)).ravel()
    if scale is None:
        pass
    elif scale == 'sqrt':
        tfs = np.sqrt(tfs)
    elif scale == 'log':
        tfs = np.log(tfs) + 1.0
    else:
        raise ValueError(
            'scale = {} is invalid; value must be one of {}'.format(
                scale, {None, 'sqrt', 'log'}))
    if normalize is True:
        return tfs / n_terms
    else:
        return tfs


def get_doc_freqs(doc_term_matrix, normalize=True):
    """
    Compute absolute or relative document frequencies for all terms in a
    document-term matrix.

    Args:
        doc_term_matrix (:class:`scipy.sparse.csr_matrix <scipy.sparse.csr_matrix`):
            M X N matrix, where M is the # of docs and N is the # of unique terms

            Note: Weighting on the terms doesn't matter! Could be 'tf' or 'tfidf'
            or 'binary' weighting, a term's doc freq will be the same
        normalize (bool): if True, return normalized doc frequencies, i.e.
            doc counts divided by the total number of docs; if False, return
            absolute doc counts

    Returns:
        :class:`numpy.ndarray`: array of absolute or relative document
        frequencies, with length equal to the # of unique terms, i.e. # of
        columns in ``doc_term_matrix``

    Raises:
        ValueError: if ``doc_term_matrix`` doesn't have any non-zero entries
    """
    if doc_term_matrix.nnz == 0:
        raise ValueError('`doc_term_matrix` must have at least 1 non-zero entry')
    n_docs, n_terms = doc_term_matrix.shape
    dfs = np.bincount(doc_term_matrix.indices, minlength=n_terms)
    if normalize is True:
        return dfs / n_docs
    else:
        return dfs


def get_inverse_doc_freqs(doc_term_matrix, type_='smooth'):
    """
    Compute inverse document frequencies for all terms in a document-term matrix,
    optionally smoothing the values, where idf = log(n_docs / dfs) + 1.0 .

    Args:
        doc_term_matrix (:class:`scipy.sparse.csr_matrix`): M X N sparse matrix,
            where M is the # of docs and N is the # of unique terms.
            The particular weighting of matrix values doesn't matter.
        type_ ({'standard', 'smooth', 'bm25'}): Type of IDF formulation to use.
            If 'standard', idfs => log(n_docs / dfs) + 1.0;
            if 'smooth', idfs => log(n_docs + 1 / dfs + 1) + 1.0, i.e. 1 is added
            to all document frequencies, equivalent to adding a single document
            to the corpus containing every unique term;
            if 'bm25', idfs => log((n_docs - dfs + 0.5) / (dfs + 0.5)), which is
            a form commonly used in BM25 ranking that allows for extremely common
            terms to have negative idf weights.

    Returns:
        :class:`numpy.ndarray`: Array of inverse document frequencies, with length
        equal to the # of unique terms (# of columns) in ``doc_term_matrix``.
    """
    dfs = get_doc_freqs(doc_term_matrix, normalize=False)
    n_docs, _ = doc_term_matrix.shape
    if type_ == 'standard':
        return np.log(n_docs / dfs) + 1.0
    elif type_ == 'smooth':
        n_docs += 1
        dfs += 1
        return np.log(n_docs / dfs) + 1.0
    elif type_ == 'bm25':
        idfs = np.log((n_docs - dfs + 0.5) / (dfs + 0.5))
    else:
        raise ValueError(
            'type_ = {} is invalid; value must be one of {}'.format(
                type_, {'standard', 'smooth', 'bm25'}))


def get_doc_lengths(doc_term_matrix, scale=None):
    """
    Compute the lengths (i.e. number of terms) for all documents in a
    document-term matrix.

    Args:
        doc_term_matrix (:class:`scipy.sparse.csr_matrix`): M X N sparse matrix,
            where M is the # of docs, N is the # of unique terms, and values are
            the absolute counts of term n per doc m.
        scale (str): Scaling applied to document lengths. If 'sqrt' or 'log',
            the square-root or natural-log of document lengths are returned.
            If None, document lengths are returned as-is.

    Returns:
        :class:`numpy.ndarray`: Array of document lengths, with length equal to
            the # of documents (i.e. # of rows) in ``doc_term_matrix``.

    Raises:
        ValueError: if ``scale`` isn't one of {None, "sqrt", "log"}.
    """
    dls = np.asarray(doc_term_matrix.sum(axis=1)).ravel()
    if scale is None:
        return dls
    elif scale == 'sqrt':
        return np.sqrt(dls)
    elif scale == 'log':
        return np.log(dls) + 1.0
    else:
        raise ValueError(
            '`scale` = {} invalid; must be one of {}'.format(
                scale, {None, 'sqrt', 'log'}))


def get_information_content(doc_term_matrix):
    """
    Compute information content for all terms in a document-term matrix. IC is a
    float in [0.0, 1.0], defined as ``-df * log2(df) - (1 - df) * log2(1 - df)``,
    where df is a term's normalized document frequency.

    Args:
        doc_term_matrix (:class:`scipy.sparse.csr_matrix <scipy.sparse.csr_matrix`):
            M X N matrix, where M is the # of docs and N is the # of unique terms

            Note: Weighting on the terms doesn't matter! Could be 'tf' or 'tfidf'
            or 'binary' weighting, a term's information content will be the same

    Returns:
        :class:`numpy.ndarray`: array of term information content values,
        with length equal to the # of unique terms, i.e. # of
        columns in ``doc_term_matrix``

    Raises:
        ValueError: if ``doc_term_matrix`` doesn't have any non-zero entries
    """
    dfs = get_doc_freqs(doc_term_matrix, normalize=True)
    ics = -dfs * np.log2(dfs) - (1 - dfs) * np.log2(1 - dfs)
    ics[np.isnan(ics)] = 0.0  # NaN values not permitted!
    return ics


def filter_terms_by_df(doc_term_matrix, term_to_id,
                       max_df=1.0, min_df=1, max_n_terms=None):
    """
    Filter out terms that are too common and/or too rare (by document frequency),
    and compactify the top ``max_n_terms`` in the ``id_to_term`` mapping accordingly.
    Borrows heavily from the ``sklearn.feature_extraction.text`` module.

    Args:
        doc_term_matrix (:class:`scipy.sparse.csr_matrix`): M X N matrix, where
            M is the # of docs and N is the # of unique terms.
        term_to_id (Dict[str, int]): Mapping of term string to unique term id,
            e.g. :attr:`Vectorizer.vocabulary`.
        min_df (float or int): if float, value is the fractional proportion of
            the total number of documents and must be in [0.0, 1.0]; if int,
            value is the absolute number; filter terms whose document frequency
            is less than ``min_df``
        max_df (float or int): if float, value is the fractional proportion of
            the total number of documents and must be in [0.0, 1.0]; if int,
            value is the absolute number; filter terms whose document frequency
            is greater than ``max_df``
        max_n_terms (int): only include terms whose *term* frequency is within
            the top `max_n_terms`

    Returns:
        :class:`scipy.sparse.csr_matrix <scipy.sparse.csr_matrix>`: sparse matrix
        of shape (# docs, # unique *filtered* terms), where value (i, j) is the
        weight of term j in doc i

        dict: id to term mapping, where keys are unique *filtered* integers as
        term ids and values are corresponding strings

    Raises:
        ValueError: if ``max_df`` or ``min_df`` or ``max_n_terms`` < 0
    """
    if max_df == 1.0 and min_df == 1 and max_n_terms is None:
        return doc_term_matrix, term_to_id
    if max_df < 0 or min_df < 0 or (max_n_terms is not None and max_n_terms < 0):
        raise ValueError('max_df, min_df, and max_n_terms may not be negative')

    n_docs, n_terms = doc_term_matrix.shape
    max_doc_count = max_df if isinstance(max_df, int) else int(max_df * n_docs)
    min_doc_count = min_df if isinstance(min_df, int) else int(min_df * n_docs)
    if max_doc_count < min_doc_count:
        raise ValueError('max_df corresponds to fewer documents than min_df')

    # calculate a mask based on document frequencies
    dfs = get_doc_freqs(doc_term_matrix, normalize=False)
    mask = np.ones(n_terms, dtype=bool)
    if max_doc_count < n_docs:
        mask &= dfs <= max_doc_count
    if min_doc_count > 1:
        mask &= dfs >= min_doc_count
    if max_n_terms is not None and mask.sum() > max_n_terms:
        tfs = get_term_freqs(doc_term_matrix, normalize=False)
        top_mask_inds = (-tfs[mask]).argsort()[:max_n_terms]
        new_mask = np.zeros(n_terms, dtype=bool)
        new_mask[np.where(mask)[0][top_mask_inds]] = True
        mask = new_mask

    # map old term indices to new ones
    new_indices = np.cumsum(mask) - 1
    term_to_id = {term: new_indices[old_index]
                  for term, old_index in term_to_id.items()
                  if mask[old_index]}

    kept_indices = np.where(mask)[0]
    if len(kept_indices) == 0:
        msg = 'After filtering, no terms remain; try a lower `min_df` or higher `max_df`'
        raise ValueError(msg)

    return (doc_term_matrix[:, kept_indices], term_to_id)


def filter_terms_by_ic(doc_term_matrix, term_to_id,
                       min_ic=0.0, max_n_terms=None):
    """
    Filter out terms that are too common and/or too rare (by information content),
    and compactify the top ``max_n_terms`` in the ``id_to_term`` mapping accordingly.
    Borrows heavily from the ``sklearn.feature_extraction.text`` module.

    Args:
        doc_term_matrix (:class:`scipy.sparse.csr_matrix`): M X N matrix, where
            M is the # of docs and N is the # of unique terms.
        term_to_id (Dict[str, int]): Mapping of term string to unique term id,
            e.g. :attr:`Vectorizer.vocabulary`.
        min_ic (float): filter terms whose information content is less than this
            value; must be in [0.0, 1.0]
        max_n_terms (int): only include terms whose information content is within
            the top ``max_n_terms``

    Returns:
        :class:`scipy.sparse.csr_matrix <scipy.sparse.csr_matrix>`: sparse matrix
        of shape (# docs, # unique *filtered* terms), where value (i, j) is the
        weight of term j in doc i

        dict: id to term mapping, where keys are unique *filtered* integers as
        term ids and values are corresponding strings

    Raises:
        ValueError: if ``min_ic`` not in [0.0, 1.0] or ``max_n_terms`` < 0
    """
    if min_ic == 0.0 and max_n_terms is None:
        return doc_term_matrix, term_to_id
    if min_ic < 0.0 or min_ic > 1.0:
        raise ValueError('min_ic must be a float in [0.0, 1.0]')
    if max_n_terms is not None and max_n_terms < 0:
        raise ValueError('max_n_terms may not be negative')

    _, n_terms = doc_term_matrix.shape

    # calculate a mask based on document frequencies
    ics = get_information_content(doc_term_matrix)
    mask = np.ones(n_terms, dtype=bool)
    if min_ic > 0.0:
        mask &= ics >= min_ic
    if max_n_terms is not None and mask.sum() > max_n_terms:
        top_mask_inds = (-ics[mask]).argsort()[:max_n_terms]
        new_mask = np.zeros(n_terms, dtype=bool)
        new_mask[np.where(mask)[0][top_mask_inds]] = True
        mask = new_mask

    # map old term indices to new ones
    new_indices = np.cumsum(mask) - 1
    term_to_id = {term: new_indices[old_index]
                  for term, old_index in term_to_id.items()
                  if mask[old_index]}

    kept_indices = np.where(mask)[0]
    if len(kept_indices) == 0:
        raise ValueError('After filtering, no terms remain; try a lower `min_ic`')

    return (doc_term_matrix[:, kept_indices], term_to_id)
