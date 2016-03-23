"""
Module to facilitate topic modeling with gensim. For example::

    >>> texts = (text for _, _, text in
    ...          textacy.corpora.wikipedia.get_plaintext_pages('enwiki-latest-pages-articles.xml.bz2', max_n_pages=1000))
    >>> proc_texts = (proc_text for proc_text in preprocess_texts(texts))
    >>> spacy_docs = (spacy_doc for spacy_doc in
    ...               texts_to_spacy_docs(proc_texts, 'en', merge_nes=True, merge_nps=False))
    >>> textacy.fileio.write_spacy_docs(spacy_docs, 'wiki_spacy_docs.bin')
    >>> spacy_docs = (spacy_doc for spacy_doc in
    ...               textacy.fileio.read_spacy_docs(textacy.data.load_spacy_pipeline().vocab, 'wiki_spacy_docs.bin'))
    >>> term_lists = (term_list for term_list in
    ...               spacy_docs_to_term_lists(spacy_docs, lemmatize=True,
    ...                                        filter_stops=True, filter_punct=True, filter_nums=False,
    ...                                        good_pos_tags=None, bad_pos_tags=None))
    >>> gcorpus, gdict = term_lists_to_gensim_corpus(
    ...     list(term_lists), min_doc_count=3, max_doc_freq=0.95, keep_top_n=50000)
    >>> topics = train_topic_model(gcorpus, gdict, 'lda', 10,
    ...                            n_top_terms=25, n_top_docs=10,
    ...                            save_to_disk=None)
"""
import logging
import numpy as np
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.externals import joblib

from textacy import data, extract, fileio, preprocess, spacy_utils


logger = logging.getLogger(__name__)


class TopicModel(object):
    """
    Args:
        model_type ({'nmf', 'lda', 'lsa'})
        n_topics (int)
        kwargs:
            see individual sklearn pages for full details

    Notes:
        - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
        - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
        - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
    """

    def __init__(self, model_type, n_topics, **kwargs):
        if model_type == 'nmf':
            self.model = NMF(
                n_components=n_topics,
                alpha=kwargs.get('alpha', 0.1),
                l1_ratio=kwargs.get('l1_ratio', 0.5),
                max_iter=kwargs.get('max_iter', 200),
                random_state=kwargs.get('random_state', 1),
                shuffle=kwargs.get('shuffle', False))
        elif model_type == 'lda':
            self.model = LatentDirichletAllocation(
                n_topics=n_topics,
                max_iter=kwargs.get('max_iter', 10),
                random_state=kwargs.get('random_state', 1),
                learning_method=kwargs.get('learning_method', 'online'),
                learning_offset=kwargs.get('learning_offset', 10.0),
                batch_size=kwargs.get('batch_size', 128),
                n_jobs=kwargs.get('n_jobs', 1))
        elif model_type == 'lsa':
            self.model = TruncatedSVD(
                n_components=n_topics,
                algorithm=kwargs.get('algorithm', 'randomized'),
                n_iter=kwargs.get('n_iter', 5),
                random_state=kwargs.get('random_state', 1))
        else:
            msg = 'model_type "{}" invalid; must be in {}'.format(
                model_type, {'nmf', 'lda', 'lsa'})
            raise ValueError(msg)
        self.model_type = model_type

    def fit(self, doc_term_matrix):
        self.model.fit(doc_term_matrix)

    def partial_fit(self, doc_term_matrix):
        if model_type == 'lda':
            self.model.partial_fit(doc_term_matrix)
        else:
            raise TypeError('only "lda" models have partial fit functionality')

    def transform(self, doc_term_matrix):
        self.model.transform(doc_term_matrix)

    def save(self, filename):
        filenames = joblib.dump(self.model, filename, compress=3)
        logger.info('{} model saved to {}'.format(self.model_type, filename))

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


#
#
# def preprocess_texts(texts):
#     """
#     Default preprocessing for raw texts before topic modeling: no URLs, emails, or
#     phone numbers, plus normalized whitespace.
#
#     Args:
#         texts (iterable(str))
#
#     Yields:
#         str: next preprocessed ``text`` in ``texts``
#     """
#     for text in texts:
#         yield preprocess.preprocess_text(
#             text, no_urls=True, no_emails=True, no_phone_numbers=True)
#
#
# def texts_to_spacy_docs(texts, lang, spacy_nlp=None
#                         merge_nes=False, merge_ncs=False):
#     """
#     Pass (preprocessed?) texts through Spacy's NLP pipeline, optionally merging
#     named entities and noun chunks into single tokens (NOTE: merging is *slow*).
#
#     Args:
#         texts (iterable(str))
#         lang (str, {'en'}): language of the input text, needed for initializing
#             a spacy nlp pipeline
#         spacy_nlp (``spacy.Language``)
#         merge_nes (bool, optional): if True, merge named entities into single tokens
#         merge_ncs (bool, optional): if True, merge noun chunks into single tokens
#
#     Yields:
#         ``spacy.Doc``: doc processed from next text in ``texts``
#     """
#     if spacy_nlp is None:
#         spacy_nlp = data.load_spacy_pipeline(
#             lang=lang, entity=merge_nes, parser=merge_ncs)
#     for spacy_doc in spacy_nlp.pipe(texts, tag=True, parse=merge_ncs, entity=merge_nes,
#                                     n_threads=2, batch_size=1000):
#         if merge_nes is True:
#             spacy_utils.merge_spans(
#                 extract.named_entities(
#                 spacy_doc, bad_ne_types='numeric', drop_determiners=True))
#         if merge_ncs is True:
#             spacy_utils.merge_spans(
#                 extract.noun_chunks(
#                     spacy_doc, drop_determiners=True))
#         yield spacy_doc


# def spacy_docs_to_term_lists(spacy_docs, lemmatize=True,
#                              filter_stops=True, filter_punct=True, filter_nums=False,
#                              good_pos_tags=None, bad_pos_tags=None):
#     """
#     Extract a (filtered) list of (lemmatized) terms as strings from each ``spacy.Doc``
#     in a sequence of docs.
#
#     Args:
#         spacy_docs (iterable(``spacy.Doc``))
#         lemmatize (bool, optional)
#         filter_stops (bool, optional)
#         filter_punct (bool, optional)
#         filter_nums (bool, optional)
#         good_pos_tags (set(str) or 'numeric', optional)
#         bad_pos_tags (set(str) or 'numeric', optional)
#
#     Yields:
#         list(str)
#     """
#     for spacy_doc in spacy_docs:
#         if lemmatize is True:
#             yield [word.lemma_ for word in
#                    extract.words(spacy_doc,
#                                  filter_stops=filter_stops, filter_punct=filter_punct, filter_nums=filter_nums,
#                                  good_pos_tags=good_pos_tags, bad_pos_tags=bad_pos_tags)]
#         else:
#             yield [word.orth_ for word in
#                    extract.words(spacy_doc,
#                                  filter_stops=filter_stops, filter_punct=filter_punct, filter_nums=filter_nums,
#                                  good_pos_tags=good_pos_tags, bad_pos_tags=bad_pos_tags)]

#
# def train_topic_model(doc_term_matrix, model_type, n_topics, save=False, **kwargs):
#     """
#     Train a topic model (NMF, LDA, or LSA) on a corpus represented as a document-
#     term matrix and return the trained model; optionally save the model to disk.
#
#     Args:
#         doc_term_matrix (array-like or sparse matrix): corpus represented as a
#             document-term matrix with shape n_docs x n_terms; NOTE: LDA expects
#             tf-weighting, while NMF and LSA may do better with tfidf-weighting!
#         model_type (str, {'nmf', 'lda', 'lsa'}): type of topic model to train
#         n_topics (int): number of topics in the model to be trained
#         save (str, optional): if not False, gives /path/to/filename where trained
#             topic model will be saved to disk using `joblib <https://pythonhosted.org/joblib/index.html>`_`
#         **kwargs:
#             model-specific keyword arguments; if not specified, default values are
#             used. See the models' scikit-learn documentation pages for more.
#
#     Returns:
#         ``sklearn.decomposition.<model>``
#
#     Raises:
#         ValueError: if ``model_type`` not in ``{'nmf', 'lda', 'lsa'}``
#
#     Notes:
#         - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
#         - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
#         - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
#     """
#     if model_type == 'nmf':
#         model = NMF(
#             n_components=n_topics, alpha=0.1, l1_ratio=0.5,
#             max_iter=kwargs.get('max_iter', 200),
#             random_state=kwargs.get('random_state', 1),
#             shuffle=kwargs.get('shuffle', False)
#             ).fit(doc_term_matrix)
#     elif model_type == 'lda':
#         model = LatentDirichletAllocation(
#             n_topics=n_topics,
#             max_iter=kwargs.get('max_iter', 10),
#             random_state=kwargs.get('random_state', 1),
#             learning_method=kwargs.get('learning_method', 'online'),
#             learning_offset=kwargs.get('learning_offset', 10.0),
#             batch_size=kwargs.get('batch_size', 128),
#             n_jobs=kwargs.get('n_jobs', 1)
#             ).fit(doc_term_matrix)
#     elif model_type == 'lsa':
#         model = TruncatedSVD(
#             n_components=n_topics, algorithm='randomized',
#             n_iter=kwargs.get('n_iter', 5),
#             random_state=kwargs.get('random_state', 1)
#             ).fit(doc_term_matrix)
#     else:
#         msg = 'model_type "{}" invalid; must be in {}'.format(
#             model_type, {'nmf', 'lda', 'lsa'})
#         raise ValueError(msg)
#
#     if save:
#         filenames = joblib.dump(model, save, compress=3)
#         logger.info('{} model saved to {}'.format(model_type, save))
#
#     return model
