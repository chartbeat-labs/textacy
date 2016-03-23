"""
Train and apply a topic model to vectorized texts. For example::

    >>> # first, stream a corpus with metadata from disk
    >>> pages = ([pid, title, text] for pid, title, text
    ...          in textacy.corpora.wikipedia.get_plaintext_pages('enwiki-latest-pages-articles.xml.bz2', max_n_pages=100))
    >>> content_stream, metadata_stream = textacy.fileio.read.split_content_and_metadata(pages, 2, itemwise=False)
    >>> metadata_stream = ({'pageid': m[0], 'title': m[1]} for m in metadata_stream)
    >>> corpus = textacy.TextCorpus.from_texts('en', content_stream, metadata=metadata_stream)
    >>> # next, tokenize and vectorize the corpus
    >>> terms_lists = (doc.as_terms_list(words=True, ngrams=False, named_entities=True)
    ...                for doc in corpus)
    >>> doc_term_matrix, id2term = corpus.as_doc_term_matrix(
    ...     terms_lists, weighting='tfidf', normalize=True, smooth_idf=True,
    ...     min_df=3, max_df=0.95, max_n_terms=100000)
    >>> # now initialize and train a topic model
    >>> model = textacy.tm.TopicModel('nmf', n_topics=20)
    >>> model.fit(doc_term_matrix)
    >>> # transform the corpus and interpret our model
    >>> doc_topic_matrix = model.transform(doc_term_matrix)
    >>> for i, top_terms in enumerate(model.get_top_topic_terms(id2term)):
    ...     print('topic {}:'.format(i), '   '.join(top_terms))
    >>> for i, docs in enumerate(model.get_top_topic_docs(doc_topic_matrix, n_docs=5)):
    ...     print('\n{}'.format(i))
    ...     for j in docs:
    ...         print(corpus[j].metadata['title'])
    >>> for i, val in enumerate(model.get_topic_weights(doc_topic_matrix)):
    ...     print(i, val)
    >>> # assess topic quality through a coherence metric
    >>> # TBD
    >>> # persist our topic model to disk
    >>> model.save('nmf-20topics.model')
"""
import logging
import numpy as np
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.externals import joblib

from textacy import data, extract, fileio, preprocess, spacy_utils
from textacy.compat import str


logger = logging.getLogger(__name__)


class TopicModel(object):
    """
    Args:
        model ({'nmf', 'lda', 'lsa'} or ``sklearn.decomposition.<model>``)
        n_topics (int, optional): number of topics in the model to be initialized
        kwargs:
            variety of parameters used to initialize the model; see individual
            sklearn pages for full details

    Raises:
        ValueError: if ``model`` not in ``{'nmf', 'lda', 'lsa'}`` or is not an
            NMF, LatentDirichletAllocation, or TruncatedSVD instance

    Notes:
        - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
        - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
        - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
    """
    def __init__(self, model, n_topics=10, **kwargs):
        if isinstance(model, (NMF, LatentDirichletAllocation, TruncatedSVD)):
            self.model = model
        else:
            self.init_model(model, n_topics=10, **kwargs)

    def init_model(self, model, n_topics=10, **kwargs):
        if model == 'nmf':
            self.model = NMF(
                n_components=n_topics,
                alpha=kwargs.get('alpha', 0.1),
                l1_ratio=kwargs.get('l1_ratio', 0.5),
                max_iter=kwargs.get('max_iter', 200),
                random_state=kwargs.get('random_state', 1),
                shuffle=kwargs.get('shuffle', False))
        elif model == 'lda':
            self.model = LatentDirichletAllocation(
                n_topics=n_topics,
                max_iter=kwargs.get('max_iter', 10),
                random_state=kwargs.get('random_state', 1),
                learning_method=kwargs.get('learning_method', 'online'),
                learning_offset=kwargs.get('learning_offset', 10.0),
                batch_size=kwargs.get('batch_size', 128),
                n_jobs=kwargs.get('n_jobs', 1))
        elif model == 'lsa':
            self.model = TruncatedSVD(
                n_components=n_topics,
                algorithm=kwargs.get('algorithm', 'randomized'),
                n_iter=kwargs.get('n_iter', 5),
                random_state=kwargs.get('random_state', 1))
        else:
            msg = 'model "{}" invalid; must be {}'.format(
                model, {'nmf', 'lda', 'lsa'})
            raise ValueError(msg)

    def save(self, filename):
        filenames = joblib.dump(self.model, filename, compress=3)
        logger.info('{} model saved to {}'.format(self.model, filename))

    @classmethod
    def load(cls, filename):
        model = joblib.load(filename)
        return cls(model, n_topics=len(model.components_))

    def fit(self, doc_term_matrix):
        self.model.fit(doc_term_matrix)

    def partial_fit(self, doc_term_matrix):
        if isinstance(self.model, LatentDirichletAllocation):
            self.model.partial_fit(doc_term_matrix)
        else:
            raise TypeError('only LatentDirichletAllocation models have partial fit')

    def transform(self, doc_term_matrix):
        return self.model.transform(doc_term_matrix)

    def get_doc_topic_matrix(self, doc_term_matrix, normalize=True):
        """
        Transform a document-term matrix into a document-topic matrix, where rows
        correspond to documents and columns to the topics in the topic model.

        Args:
            doc_term_matrix (array-like or sparse matrix): corpus represented as a
                document-term matrix with shape (n_docs, n_terms); NOTE: LDA expects
                tf-weighting, while NMF and LSA may do better with tfidf-weighting!
            normalize (bool, optional): if True, the values in each row are normalized,
                i.e. topic weights on each document sum to 1

        Returns:
            ``numpy.ndarray``: document-topic matrix with shape (n_docs, n_topics)
        """
        doc_topic_matrix = self.transform(doc_term_matrix)
        if normalize is True:
            return doc_topic_matrix / np.sum(doc_topic_matrix, axis=1, keepdims=True)
        else:
            return doc_topic_matrix

    def get_top_topic_terms(self, vocab, n_topics=-1, n_terms=10, weights=False):
        """
        Get the top ``n_terms`` terms by weight per topic in ``model``.

        Args:
            vocab (list(str) or dict): object that returns the term string corresponding
                to term id ``i`` through ``feature_names[i]``; could be a list of strings
                where the index represents the term id, such as that returned by
                ``sklearn.feature_extraction.text.CountVectorizer.get_feature_names()``,
                or a mapping of term id: term string
            n_topics (int, optional): number of topics for which to return top terms;
                if -1, all topics' terms are returned
            n_terms (int, optional): number of top terms to return per topic
            weights (bool, optional): if True, terms are returned with their corresponding
                topic weights; otherwise, terms are returned without weights

        Returns:
            list(list(str)) or list(list((str, float))):
                the ith list of terms or (term, weight) tuples corresponds to topic i
        """
        if n_topics == -1:
            n_topics = len(self.model.components_)
        if weights is False:
            return [[vocab[i] for i in np.argsort(topic)[:-n_terms - 1:-1]]
                    for topic in self.model.components_[:n_topics]]
        else:
            return [[(vocab[i], topic[i]) for i in np.argsort(topic)[:-n_terms - 1:-1]]
                    for topic in self.model.components_[:n_topics]]

    def get_top_topic_docs(self, doc_topic_matrix,
                           n_topics=-1, n_docs=10, weights=False):
        """
        Get the top ``n_docs`` docs by weight per topic in ``doc_topic_matrix``.

        Args:
            doc_topic_matrix (numpy.ndarray): document-topic matrix with shape
                (n_docs, n_topics), the result of calling
                :func:`get_doc_topic_matrix() <textacy.topic_modeling.get_doc_topic_matrix>`
            n_topics (int, optional): number of topics for which to return top docs;
                if -1, all topics' docs are returned
            n_docs (int, optional): number of top docs to return per topic
            weights (bool, optional): if True, docs are returned with their corresponding
                (normalized) topic weights; otherwise, docs are returned without weights

        Returns:
            list(list(str)) or list(list((str, float))): the ith list of docs or
                (doc, weight) tuples corresponds to topic i
        """
        if n_topics == -1:
            n_topics = doc_topic_matrix.shape[1]

        if weights is False:
            return [[doc_idx
                     for doc_idx in np.argsort(doc_topic_matrix[:, i])[:-n_docs - 1:-1]]
                    for i in range(min(doc_topic_matrix.shape[1], n_topics))]
        else:
            return [[(doc_idx, doc_topic_matrix[doc_idx, i])
                     for doc_idx in np.argsort(doc_topic_matrix[:, i])[:-n_docs - 1:-1]]
                    for i in range(min(doc_topic_matrix.shape[1], n_topics))]

    def get_top_doc_topics(self, doc_topic_matrix, n_docs=-1, n_topics=3, weights=False):
        """
        Get the top ``n_topics`` topics by weight per doc in ``doc_topic_matrix``.

        Args:
            doc_topic_matrix (numpy.ndarray): document-topic matrix with shape
                (n_docs, n_topics), the result of calling
                :func:`get_doc_topic_matrix() <textacy.topic_modeling.get_doc_topic_matrix>`
            n_docs (int, optional): number of docs for which to return top topics;
                if -1, all docs' top topics are returned
            n_topics (int, optional): number of top topics to return per doc
            weights (bool, optional): if True, docs are returned with their corresponding
                (normalized) topic weights; otherwise, docs are returned without weights

        Returns:
            list(list(str)) or list(list((str, float))): the ith list of topics or
                (topic, weight) tuples corresponds to doc i
        """
        if n_docs == -1:
            n_docs = doc_topic_matrix.shape[0]

        if weights is False:
            return [[topic_idx
                     for topic_idx in np.argsort(doc_topic_matrix[i, :])[:-n_topics - 1:-1]]
                    for i in range(min(doc_topic_matrix.shape[0], n_docs))]
        else:
            return [[(topic_idx, doc_topic_matrix[i, topic_idx])
                     for topic_idx in np.argsort(doc_topic_matrix[i, :])[:-n_topics - 1:-1]]
                    for i in range(min(doc_topic_matrix.shape[0], n_docs))]

    def get_topic_weights(self, doc_topic_matrix):
        """
        Get the overall weight of topics across an entire corpus. Note: Values depend
        on whether topic weights per document in ``doc_topic_matrix`` were normalized,
        or not. I suppose either way makes sense... o_O

        Args:
            doc_topic_matrix (numpy.ndarray): document-topic matrix with shape
                (n_docs, n_topics), the result of calling
                :func:`get_doc_topic_matrix() <textacy.topic_modeling.get_doc_topic_matrix>`

        Returns:
            ``numpy.ndarray``: the ith element is the ith topic's overall weight
        """
        return doc_topic_matrix.sum(axis=0) / doc_topic_matrix.sum(axis=0).sum()

    def get_topic_coherence(self, topic_idx):
        raise NotImplementedError()

    def get_model_coherence(self):
        raise NotImplementedError()
