"""
Topic Models
------------

Convenient and consolidated topic-modeling, built on ``scikit-learn``.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import joblib
import numpy as np
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD

from .. import compat
from .. import viz

LOGGER = logging.getLogger(__name__)


class TopicModel(object):
    """
    Train and apply a topic model to vectorized texts using scikit-learn's
    implementations of LSA, LDA, and NMF models. Inspect and visualize results.
    Save and load trained models to and from disk.

    Prepare a vectorized corpus (i.e. document-term matrix) and corresponding
    vocabulary (i.e. mapping of term strings to column indices in the matrix).
    See :class:`textacy.vsm.Vectorizer` for details. In short::

        >>> vectorizer = Vectorizer(
        ...     tf_type='linear', apply_idf=True, idf_type='smooth', norm='l2',
        ...     min_df=3, max_df=0.95, max_n_terms=100000)
        >>> doc_term_matrix = vectorizer.fit_transform(terms_list)

    Initialize and train a topic model::

        >>> model = textacy.tm.TopicModel('nmf', n_topics=20)
        >>> model.fit(doc_term_matrix)
        >>> model
        TopicModel(n_topics=10, model=NMF)

    Transform the corpus and interpret our model::

        >>> doc_topic_matrix = model.transform(doc_term_matrix)
        >>> for topic_idx, top_terms in model.top_topic_terms(vectorizer.id_to_term, topics=[0,1]):
        ...     print('topic', topic_idx, ':', '   '.join(top_terms))
        topic 0 : people   american   go   year   work   think   $   today   money   america
        topic 1 : rescind   quorum   order   unanimous   consent   ask   president   mr.   madam   absence
        >>> for topic_idx, top_docs in model.top_topic_docs(doc_topic_matrix, topics=[0,1], top_n=2):
        ...     print(topic_idx)
        ...     for j in top_docs:
        ...         print(corpus[j].metadata['title'])
        0
        THE MOST IMPORTANT ISSUES FACING THE AMERICAN PEOPLE
        55TH ANNIVERSARY OF THE BATTLE OF CRETE
        1
        CHEMICAL WEAPONS CONVENTION
        MFN STATUS FOR CHINA
        >>> for doc_idx, topics in model.top_doc_topics(doc_topic_matrix, docs=range(5), top_n=2):
        ...     print(corpus[doc_idx].metadata['title'], ':', topics)
        JOIN THE SENATE AND PASS A CONTINUING RESOLUTION : (9, 0)
        MEETING THE CHALLENGE : (2, 0)
        DISPOSING OF SENATE AMENDMENT TO H.R. 1643, EXTENSION OF MOST-FAVORED- NATION TREATMENT FOR BULGARIA : (0, 9)
        EXAMINING THE SPEAKER'S UPCOMING TRAVEL SCHEDULE : (0, 9)
        FLOODING IN PENNSYLVANIA : (0, 9)
        >>> for i, val in enumerate(model.topic_weights(doc_topic_matrix)):
        ...     print(i, val)
        0 0.302796022302
        1 0.0635617650602
        2 0.0744927472417
        3 0.0905778808867
        4 0.0521162262192
        5 0.0656303769725
        6 0.0973516532757
        7 0.112907245542
        8 0.0680659204364
        9 0.0725001620636

    Visualize the model::

        >>> model.termite_plot(doc_term_matrix, vectorizer.id_to_term,
        ...                    topics=-1,  n_terms=25, sort_terms_by='seriation')

    Persist our topic model to disk::

        >>> model.save('nmf-10topics.pkl')

    Args:
        model ({'nmf', 'lda', 'lsa'} or ``sklearn.decomposition.<model>``)
        n_topics (int): number of topics in the model to be initialized
        **kwargs:
            variety of parameters used to initialize the model; see individual
            sklearn pages for full details

    Raises:
        ValueError: if ``model`` not in ``{'nmf', 'lda', 'lsa'}`` or is not an
            NMF, LatentDirichletAllocation, or TruncatedSVD instance

    See Also:
        - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
        - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
        - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
    """

    def __init__(self, model, n_topics=10, **kwargs):
        if isinstance(model, (NMF, LatentDirichletAllocation, TruncatedSVD)):
            self.model = model
        else:
            self.init_model(model, n_topics=n_topics, **kwargs)

    def init_model(self, model, n_topics=10, **kwargs):
        if model == "nmf":
            self.model = NMF(
                n_components=n_topics,
                alpha=kwargs.get("alpha", 0.1),
                l1_ratio=kwargs.get("l1_ratio", 0.5),
                max_iter=kwargs.get("max_iter", 200),
                random_state=kwargs.get("random_state", 1),
                shuffle=kwargs.get("shuffle", False),
            )
        elif model == "lda":
            self.model = LatentDirichletAllocation(
                n_topics=n_topics,
                max_iter=kwargs.get("max_iter", 10),
                random_state=kwargs.get("random_state", 1),
                learning_method=kwargs.get("learning_method", "online"),
                learning_offset=kwargs.get("learning_offset", 10.0),
                batch_size=kwargs.get("batch_size", 128),
                n_jobs=kwargs.get("n_jobs", 1),
            )
        elif model == "lsa":
            self.model = TruncatedSVD(
                n_components=n_topics,
                algorithm=kwargs.get("algorithm", "randomized"),
                n_iter=kwargs.get("n_iter", 5),
                random_state=kwargs.get("random_state", 1),
            )
        else:
            msg = 'model "{}" invalid; must be {}'.format(model, {"nmf", "lda", "lsa"})
            raise ValueError(msg)

    def __repr__(self):
        return "TopicModel(n_topics={}, model={})".format(
            self.n_topics, str(self.model).split("(", 1)[0]
        )

    def save(self, filepath):
        _ = joblib.dump(self.model, filepath, compress=3)
        LOGGER.info("%s model saved to %s", self.model, filepath)

    @classmethod
    def load(cls, filepath):
        model = joblib.load(filepath)
        n_topics = model.n_topics if hasattr(model, "n_topics") else model.n_components
        return cls(model, n_topics=n_topics)

    def fit(self, doc_term_matrix):
        self.model.fit(doc_term_matrix)

    def partial_fit(self, doc_term_matrix):
        if isinstance(self.model, LatentDirichletAllocation):
            self.model.partial_fit(doc_term_matrix)
        else:
            raise TypeError("only LatentDirichletAllocation models have partial_fit")

    def transform(self, doc_term_matrix):
        return self.model.transform(doc_term_matrix)

    @property
    def n_topics(self):
        try:
            return self.model.n_topics
        except AttributeError:
            return self.model.n_components

    def get_doc_topic_matrix(self, doc_term_matrix, normalize=True):
        """
        Transform a document-term matrix into a document-topic matrix, where rows
        correspond to documents and columns to the topics in the topic model.

        Args:
            doc_term_matrix (array-like or sparse matrix): Corpus represented as a
                document-term matrix with shape (n_docs, n_terms). LDA expects
                tf-weighting, while NMF and LSA may do better with tfidf-weighting.
            normalize (bool): if True, the values in each row are normalized,
                i.e. topic weights on each document sum to 1

        Returns:
            :class:`numpy.ndarray`: Document-topic matrix with shape (n_docs, n_topics).
        """
        doc_topic_matrix = self.transform(doc_term_matrix)
        if normalize is True:
            return doc_topic_matrix / np.sum(doc_topic_matrix, axis=1, keepdims=True)
        else:
            return doc_topic_matrix

    def top_topic_terms(self, id2term, topics=-1, top_n=10, weights=False):
        """
        Get the top ``top_n`` terms by weight per topic in ``model``.

        Args:
            id2term (list(str) or dict): object that returns the term string corresponding
                to term id ``i`` through ``id2term[i]``; could be a list of strings
                where the index represents the term id, such as that returned by
                ``sklearn.feature_extraction.text.CountVectorizer.get_feature_names()``,
                or a mapping of term id: term string
            topics (int or Sequence[int]): topic(s) for which to return top terms;
                if -1 (default), all topics' terms are returned
            top_n (int): number of top terms to return per topic
            weights (bool): if True, terms are returned with their corresponding
                topic weights; otherwise, terms are returned without weights

        Yields:
            Tuple[int, Tuple[str]] or Tuple[int, Tuple[Tuple[str, float]]]:
                next tuple corresponding to a topic; the first element is the topic's
                index; if ``weights`` is False, the second element is a tuple of str
                representing the top ``top_n`` related terms; otherwise, the second
                is a tuple of (str, float) pairs representing the top ``top_n``
                related terms and their associated weights wrt the topic; for example::

                    >>> list(TopicModel.top_topic_terms(id2term, topics=(0, 1), top_n=2, weights=False))
                    [(0, ('foo', 'bar')), (1, ('bat', 'baz'))]
                    >>> list(TopicModel.top_topic_terms(id2term, topics=0, top_n=2, weights=True))
                    [(0, (('foo', 0.1415), ('bar', 0.0986)))]
        """
        if topics == -1:
            topics = compat.range_(self.n_topics)
        elif isinstance(topics, int):
            topics = (topics,)

        for topic_idx in topics:
            topic = self.model.components_[topic_idx]
            if weights is False:
                yield (
                    topic_idx,
                    tuple(id2term[i] for i in np.argsort(topic)[: -top_n - 1 : -1]),
                )
            else:
                yield (
                    topic_idx,
                    tuple(
                        (id2term[i], topic[i])
                        for i in np.argsort(topic)[: -top_n - 1 : -1]
                    ),
                )

    def top_topic_docs(self, doc_topic_matrix, topics=-1, top_n=10, weights=False):
        """
        Get the top ``top_n`` docs by weight per topic in ``doc_topic_matrix``.

        Args:
            doc_topic_matrix (:class:`numpy.ndarray`): document-topic matrix with shape
                (n_docs, n_topics), the result of calling :meth:`TopicModel.get_doc_topic_matrix()`
            topics (int or Sequence[int]): topic(s) for which to return top docs;
                if -1, all topics' docs are returned
            top_n (int): number of top docs to return per topic
            weights (bool): if True, docs are returned with their corresponding
                (normalized) topic weights; otherwise, docs are returned without weights

        Yields:
            Tuple[int, Tuple[int]] or Tuple[int, Tuple[Tuple[int, float]]]:
                next tuple corresponding to a topic; the first element is the topic's
                index; if ``weights`` is False, the second element is a tuple of ints
                representing the top ``top_n`` related docs; otherwise, the second
                is a tuple of (int, float) pairs representing the top ``top_n``
                related docs and their associated weights wrt the topic; for example::

                    >>> list(TopicModel.top_doc_terms(dtm, topics=(0, 1), top_n=2, weights=False))
                    [(0, (4, 2)), (1, (1, 3))]
                    >>> list(TopicModel.top_doc_terms(dtm, topics=0, top_n=2, weights=True))
                    [(0, ((4, 0.3217), (2, 0.2154)))]
        """
        if topics == -1:
            topics = compat.range_(self.n_topics)
        elif isinstance(topics, int):
            topics = (topics,)

        for topic_idx in topics:
            top_doc_idxs = np.argsort(doc_topic_matrix[:, topic_idx])[: -top_n - 1 : -1]
            if weights is False:
                yield (topic_idx, tuple(doc_idx for doc_idx in top_doc_idxs))
            else:
                yield (
                    topic_idx,
                    tuple(
                        (doc_idx, doc_topic_matrix[doc_idx, topic_idx])
                        for doc_idx in top_doc_idxs
                    ),
                )

    def top_doc_topics(self, doc_topic_matrix, docs=-1, top_n=3, weights=False):
        """
        Get the top ``top_n`` topics by weight per doc for ``docs`` in ``doc_topic_matrix``.

        Args:
            doc_topic_matrix (:class:`numpy.ndarray`): document-topic matrix with shape
                (n_docs, n_topics), the result of calling :meth:`TopicModel.get_doc_topic_matrix()`
            docs (int or Sequence[int]): docs for which to return top topics;
                if -1, all docs' top topics are returned
            top_n (int): number of top topics to return per doc
            weights (bool): if True, docs are returned with their corresponding
                (normalized) topic weights; otherwise, docs are returned without weights

        Yields:
            Tuple[int, Tuple[int]] or Tuple[int, Tuple[Tuple[int, float]]]:
                next tuple corresponding to a doc; the first element is the doc's
                index; if ``weights`` is False, the second element is a tuple of ints
                representing the top ``top_n`` related topics; otherwise, the second
                is a tuple of (int, float) pairs representing the top ``top_n``
                related topics and their associated weights wrt the doc; for example::

                    >>> list(TopicModel.top_doc_topics(dtm, docs=(0, 1), top_n=2, weights=False))
                    [(0, (1, 4)), (1, (3, 2))]
                    >>> list(TopicModel.top_doc_topics(dtm, docs=0, top_n=2, weights=True))
                    [(0, ((1, 0.2855), (4, 0.2412)))]
        """
        if docs == -1:
            docs = compat.range_(doc_topic_matrix.shape[0])
        elif isinstance(docs, int):
            docs = (docs,)

        for doc_idx in docs:
            top_topic_idxs = np.argsort(doc_topic_matrix[doc_idx, :])[: -top_n - 1 : -1]
            if weights is False:
                yield (doc_idx, tuple(topic_idx for topic_idx in top_topic_idxs))
            else:
                yield (
                    doc_idx,
                    tuple(
                        (topic_idx, doc_topic_matrix[doc_idx, topic_idx])
                        for topic_idx in top_topic_idxs
                    ),
                )

    def topic_weights(self, doc_topic_matrix):
        """
        Get the overall weight of topics across an entire corpus. Note: Values depend
        on whether topic weights per document in ``doc_topic_matrix`` were normalized,
        or not. I suppose either way makes sense... o_O

        Args:
            doc_topic_matrix (:class:`numpy.ndarray`): document-topic matrix with shape
                (n_docs, n_topics), the result of calling :meth:`TopicModel.get_doc_topic_matrix()`

        Returns:
            :class:`numpy.ndarray`: the ith element is the ith topic's overall weight
        """
        return doc_topic_matrix.sum(axis=0) / doc_topic_matrix.sum(axis=0).sum()

    # def get_topic_coherence(self, topic_idx):
    #     raise NotImplementedError()
    #
    # def get_model_coherence(self):
    #     raise NotImplementedError()

    def termite_plot(
        self,
        doc_term_matrix,
        id2term,
        topics=-1,
        sort_topics_by="index",
        highlight_topics=None,
        n_terms=25,
        rank_terms_by="topic_weight",
        sort_terms_by="seriation",
        save=False,
    ):
        """
        Make a "termite" plot for assessing topic models using a tabular layout
        to promote comparison of terms both within and across topics.

        Args:
            doc_term_matrix (:class:`numpy.ndarray` or sparse matrix): corpus
                represented as a document-term matrix with shape (n_docs, n_terms);
                may have tf- or tfidf-weighting
            id2term (List[str] or dict): object that returns the term string corresponding
                to term id ``i`` through ``id2term[i]``; could be a list of strings
                where the index represents the term id, such as that returned by
                ``sklearn.feature_extraction.text.CountVectorizer.get_feature_names()``,
                or a mapping of term id: term string
            topics (int or Sequence[int]): topic(s) to include in termite plot;
                if -1, all topics are included
            sort_topics_by ({'index', 'weight'}):
            highlight_topics (int or Sequence[int]): indices for up to 6 topics
                to visually highlight in the plot with contrasting colors
            n_terms (int): number of top terms to include in termite plot
            rank_terms_by ({'topic_weight', 'corpus_weight'}): value used
                to rank terms; the top-ranked ``n_terms`` are included in the plot
            sort_terms_by ({'seriation', 'weight', 'index', 'alphabetical'}):
                method used to vertically sort the selected top ``n_terms`` terms;
                the default ("seriation") groups similar terms together, which
                facilitates cross-topic assessment
            save (str): give the full /path/to/fname on disk to save figure

        Returns:
            ``matplotlib.axes.Axes.axis``: Axis on which termite plot is plotted.

        Raises:
            ValueError: if more than 6 topics are selected for highlighting, or
                an invalid value is passed for the sort_topics_by, rank_terms_by,
                and/or sort_terms_by params

        References:
            - Chuang, Jason, Christopher D. Manning, and Jeffrey Heer. "Termite:
              Visualization techniques for assessing textual topic models."
              Proceedings of the International Working Conference on Advanced
              Visual Interfaces. ACM, 2012.
            - for sorting by "seriation", see https://arxiv.org/abs/1406.5370

        See Also:
            :func:`viz.termite_plot <textacy.viz.termite.termite_plot>`

        TODO: `rank_terms_by` other metrics, e.g. topic salience or relevance
        """
        if highlight_topics is not None:
            if isinstance(highlight_topics, int):
                highlight_topics = (highlight_topics,)
            elif len(highlight_topics) > 6:
                raise ValueError("no more than 6 topics may be highlighted at once")

        # get topics indices
        if topics == -1:
            topic_inds = tuple(compat.range_(self.n_topics))
        elif isinstance(topics, int):
            topic_inds = (topics,)
        else:
            topic_inds = tuple(topics)

        # get topic indices in sorted order
        if sort_topics_by == "index":
            topic_inds = sorted(topic_inds)
        elif sort_topics_by == "weight":
            topic_inds = tuple(
                topic_ind
                for topic_ind in np.argsort(
                    self.topic_weights(self.transform(doc_term_matrix))
                )[::-1]
                if topic_ind in topic_inds
            )
        else:
            msg = "invalid sort_topics_by value; must be in {}".format(
                {"index", "weight"}
            )
            raise ValueError(msg)

        # get column index of any topics to highlight in termite plot
        if highlight_topics is not None:
            highlight_cols = tuple(
                i
                for i in compat.range_(len(topic_inds))
                if topic_inds[i] in highlight_topics
            )
        else:
            highlight_cols = None

        # get top term indices
        if rank_terms_by == "corpus_weight":
            term_inds = np.argsort(np.ravel(doc_term_matrix.sum(axis=0)))[
                : -n_terms - 1 : -1
            ]
        elif rank_terms_by == "topic_weight":
            term_inds = np.argsort(self.model.components_.sum(axis=0))[
                : -n_terms - 1 : -1
            ]
        else:
            msg = "invalid rank_terms_by value; must be in {}".format(
                {"corpus_weight", "topic_weight"}
            )
            raise ValueError(msg)

        # get top term indices in sorted order
        if sort_terms_by == "weight":
            pass
        elif sort_terms_by == "index":
            term_inds = sorted(term_inds)
        elif sort_terms_by == "alphabetical":
            term_inds = sorted(term_inds, key=lambda x: id2term[x])
        elif sort_terms_by == "seriation":
            topic_term_weights_mat = np.array(
                np.array(
                    [
                        self.model.components_[topic_ind][term_inds]
                        for topic_ind in topic_inds
                    ]
                )
            ).T
            # calculate similarity matrix
            topic_term_weights_sim = np.dot(
                topic_term_weights_mat, topic_term_weights_mat.T
            )
            # substract minimum of sim mat in order to keep sim mat nonnegative
            topic_term_weights_sim = (
                topic_term_weights_sim - topic_term_weights_sim.min()
            )
            # compute Laplacian matrice and its 2nd eigenvector
            L = np.diag(sum(topic_term_weights_sim, 1)) - topic_term_weights_sim
            D, V = np.linalg.eigh(L)
            D = D[np.argsort(D)]
            V = V[:, np.argsort(D)]
            fiedler = V[:, 1]
            # get permutation corresponding to sorting the 2nd eigenvector
            term_inds = [term_inds[i] for i in np.argsort(fiedler)]
        else:
            msg = "invalid sort_terms_by value; must be in {}".format(
                {"weight", "index", "alphabetical", "seriation"}
            )
            raise ValueError(msg)

        # get topic and term labels
        topic_labels = tuple("topic {}".format(topic_ind) for topic_ind in topic_inds)
        term_labels = tuple(id2term[term_ind] for term_ind in term_inds)

        # get topic-term weights to size dots
        term_topic_weights = np.array(
            [self.model.components_[topic_ind][term_inds] for topic_ind in topic_inds]
        ).T

        return viz.draw_termite_plot(
            term_topic_weights,
            topic_labels,
            term_labels,
            highlight_cols=highlight_cols,
            save=save,
        )
