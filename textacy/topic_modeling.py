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
import numpy as np
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.externals import joblib

from textacy import data, extract, fileio, preprocess, spacy_utils


logger = logging.getLogger(__name__)

# TODO: what to do about lang?


def preprocess_texts(texts):
    """
    Default preprocessing for raw texts before topic modeling: no URLs, emails, or
    phone numbers, plus normalized whitespace.

    Args:
        texts (iterable(str))

    Yields:
        str: next preprocessed ``text`` in ``texts``
    """
    for text in texts:
        yield preprocess.preprocess_text(
            text, no_urls=True, no_emails=True, no_phone_numbers=True)


def texts_to_spacy_docs(texts, lang, merge_nes=False, merge_ncs=False):
    """
    Pass (preprocessed) texts through Spacy's NLP pipeline, optionally merging
    named entities and noun chunks into single tokens (NOTE: merging is *slow*).

    Args:
        texts (iterable(str))
        lang (str, {'en'}): language of the input text, needed for initializing
            a spacy nlp pipeline
        merge_nes (bool, optional): if True, merge named entities into single tokens
        merge_ncs (bool, optional): if True, merge noun chunks into single tokens

    Yields:
        ``spacy.Doc``: doc processed from next text in ``texts``
    """
    spacy_nlp = data.load_spacy_pipeline(
        lang=lang, entity=merge_nes, parser=merge_ncs)
    for spacy_doc in spacy_nlp.pipe(texts, tag=True, parse=merge_ncs, entity=merge_nes,
                                    n_threads=2, batch_size=1000):
        if merge_nes is True:
            spacy_utils.merge_spans(
                extract.named_entities(
                spacy_doc, bad_ne_types='numeric', drop_determiners=False))
        if merge_ncs is True:
            spacy_utils.merge_spans(
                extract.noun_chunks(
                    spacy_doc, drop_determiners=False))
        yield spacy_doc


def spacy_docs_to_term_lists(spacy_docs, lemmatize=True,
                             filter_stops=True, filter_punct=True, filter_nums=False,
                             good_pos_tags=None, bad_pos_tags=None):
    """
    Extract a (filtered) list of (lemmatized) terms as strings from each ``spacy.Doc``
    in a sequence of docs.

    Args:
        spacy_docs (iterable(``spacy.Doc``))
        lemmatize (bool, optional)
        filter_stops (bool, optional)
        filter_punct (bool, optional)
        filter_nums (bool, optional)
        good_pos_tags (set(str) or 'numeric', optional)
        bad_pos_tags (set(str) or 'numeric', optional)

    Yields:
        list(str)
    """
    for spacy_doc in spacy_docs:
        if lemmatize is True:
            yield [word.lemma_ for word in
                   extract.words(spacy_doc,
                                 filter_stops=filter_stops, filter_punct=filter_punct, filter_nums=filter_nums,
                                 good_pos_tags=good_pos_tags, bad_pos_tags=bad_pos_tags)]
        else:
            yield [word.orth_ for word in
                   extract.words(spacy_doc,
                                 filter_stops=filter_stops, filter_punct=filter_punct, filter_nums=filter_nums,
                                 good_pos_tags=good_pos_tags, bad_pos_tags=bad_pos_tags)]


def term_lists_to_gensim_corpus(term_lists,
                                min_doc_count=1, max_doc_freq=1.0, keep_top_n=False):
    """
    Create a gensim corpus and accompanying dictionary from a sequence of term lists,
    optionally filtering terms by low/high frequency. NOTE: if terms are to be filtered,
    ``term_lists`` must be iterated over *twice*; if not, only a single pass is needed.

    Args:
        term_lists (iterable(list(str)))
        min_doc_count (int, optional)
        max_doc_freq (float, optional): must be in interval (0.0, 1.0]
        keep_top_n (int, optional)

    Returns:
        list((int, int))
        `gensim.Dictonary <gensim.corpora.dictionary.Dictionary`
    """
    if min_doc_count > 1 or max_doc_freq < 1.0 or keep_top_n > 0:
        # this means we have to filter the dictionary, which means we have to iterate
        # over the term_lists twice: once to build the dictionary, once to build the corpus
        gdict = GensimDictionary(documents=term_lists)
        filter_params = {}
        if min_doc_count > 1:
            filter_params['no_below'] = min_doc_count
        if max_doc_freq < 1.0:
            filter_params['no_above'] = max_doc_freq
        if keep_top_n > 0:
            filter_params['keep_n'] = keep_top_n
        gdict.filter_extremes(**filter_params)
        return ([gdict.doc2bow(term_list, allow_update=False) for term_list in term_lists], gdict)
    else:
        gdict = GensimDictionary()
        return ([gdict.doc2bow(term_list, allow_update=True) for term_list in term_lists], gdict)


def weight_gensim_corpus(corpus, weighting='tfidf'):
    """
    Weight the words in a bag-of-words (tf-weighted) corpus by either term frequency
    inverse document frequency ('tfidf') or log entropy ('logent').

    Args:
        corpus (list((int, int)))
        weighting (str, {'tfidf', 'logent'}, optional)

    Returns:
        corpus (list((int, float)))

    Raises:
        ValueError: if weighting not in {'tfidf', 'logent'}
    """
    if weighting == 'tfidf':
        tfidf_model = gensim.models.TfidfModel(corpus, normalize=True)
        corpus = tfidf_model[corpus]
    elif weighting == 'logent':
        logent_model = gensim.models.LogEntropyModel(corpus, normalize=True)
        corpus = logent_model[corpus]
    else:
        msg = 'weighting {} not valid; choose from {}'.format(weighting, {'tfidf', 'logent'})
        raise ValueError(msg)
    return corpus


def optimize_n_topics():
    raise NotImplementedError('burton is working on it...')


def train_topic_model(doc_term_matrix, model_type, n_topics,
                      save=False, **kwargs):
    """
    Train a topic model (NMF, LDA, or LSA) on a corpus represented as a document-
    term matrix and return the trained model; optionally save the model to disk.

    Args:
        doc_term_matrix (array-like or sparse matrix): corpus represented as a
            document-term matrix with shape n_docs x n_terms; NOTE: LDA expects
            tf-weighting, while NMF and LSA may do better with tfidf-weighting!
        model_type (str, {'nmf', 'lda', 'lsa'}): type of topic model to train
        n_topics (int): number of topics in the model to be trained
        save (str, optional): if not False, gives /path/to/filename where trained
            topic model will be saved to disk using `joblib <https://pythonhosted.org/joblib/index.html>`_`
        **kwargs:
            model-specific keyword arguments; if not specified, default values are
            used; see the models' scikit-learn documentation

    Returns:
        ``sklearn.decomposition.<model>``

    Raises:
        ValueError: if ``model_type`` not in ``{'nmf', 'lda', 'lsa'}``

    Notes:
        - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
        - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
        - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
    """
    if model_type == 'nmf':
        model = NMF(
            n_components=n_topics, alpha=0.1, l1_ratio=0.5,
            max_iter=kwargs.get('max_iter', 200),
            random_state=kwargs.get('random_state', 1),
            shuffle=kwargs.get('shuffle', False)
            ).fit(doc_term_matrix)
    elif model_type == 'lda':
        model = LatentDirichletAllocation(
            n_topics=n_topics,
            max_iter=kwargs.get('max_iter', 10),
            random_state=kwargs.get('random_state', 1),
            learning_method=kwargs.get('learning_method', 'online'),
            learning_offset=kwargs.get('learning_offset', 10.0),
            batch_size=kwargs.get('batch_size', 128),
            n_jobs=kwargs.get('n_jobs', 1)
            ).fit(doc_term_matrix)
    elif model_type == 'lsa':
        model = TruncatedSVD(
            n_components=n_topics, algorithm='randomized',
            n_iter=kwargs.get('n_iter', 5),
            random_state=kwargs.get('random_state', 1)
            ).fit(doc_term_matrix)
    else:
        msg = 'model_type "{}" invalid; must be in {}'.format(
            model_type, {'nmf', 'lda', 'lsa'})
        raise ValueError(msg)

    if save:
        filenames = joblib.dump(model, save, compress=3)
        logger.info('{} model saved to {}'.format(model_type, save))

    return model


def get_top_topic_terms(model, vocab, n_topics=-1, n_terms=10, weights=False):
    """
    Get the top ``n_terms`` terms by weight per topic in ``model``.

    Args:
        model (``sklearn.decomposition.<model>``)
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
        n_topics = len(model.components_)
    if weights is False:
        return [[vocab[i] for i in np.argsort(topic)[:-n_terms - 1:-1]]
                for topic in model.components_[:n_topics]]
    else:
        return [[(vocab[i], topic[i]) for i in np.argsort(topic)[:-n_terms - 1:-1]]
                for topic in model.components_[:n_topics]]


def get_top_topic_docs(model, doc_term_matrix, n_topics=-1, n_docs=10, weights=False):
    """
    Get the top ``n_docs`` docs by weight per topic in ``model``. Documents in
    ``doc_term_matrix`` are transformed by the trained ``model`` into topic space,
    normalizing such that topic contributions per doc sum to 1.

    Args:
        model (``sklearn.decomposition.<model>``)
        doc_term_matrix (array-like or sparse matrix): corpus represented as a
            document-term matrix with shape n_docs x n_terms; NOTE: LDA expects
            tf-weighting, while NMF and LSA may do better with tfidf-weighting!
        n_topics (int, optional): number of topics for which to return top docs;
            if -1, all topics' docs are returned
        n_docs (int, optional): number of top docs to return per topic
        weights (bool, optional): if True, docs are returned with their corresponding
            (normalized) topic weights; otherwise, docs are returned without weights

    Returns:
        list(list(str)) or list(list((str, float))):
            the ith list of docs or (doc, weight) tuples corresponds to topic i
    """
    if n_topics == -1:
        n_topics = len(model.components_)
    doc_topic_distr = model.transform(doc_term_matrix)
    doc_topic_distr = doc_topic_distr / np.sum(doc_topic_distr, axis=1, keepdims=True)
    if weights is False:
        return [[doc_idx
                 for doc_idx in np.argsort(doc_topic_distr[:, i])[:-n_docs - 1:-1]]
                for i, _ in enumerate(model.components_[:n_topics])]
    else:
        return [[(doc_idx, doc_topic_distr[doc_idx, i])
                 for doc_idx in np.argsort(doc_topic_distr[:, i])[:-n_docs - 1:-1]]
                for i, _ in enumerate(model.components_[:n_topics])]



#
# # get *normalized* topic weights in corpus
# topic_weights = [sum(doc_topics[:,i]) for i in range(n_topics)]
# topic_weights = [tw / sum(topic_weights) for tw in topic_weights]
# # get top documents per topic
# topic_top_docs = [list(np.argsort(doc_topics[:,i])[::-1][:n_top_docs])
#                   for i in range(n_topics)]
#
# return [{'topic_id': i,
#          'topic_weight': topic_weights[i],
#          'key_terms': topic_top_terms[i],
#          'key_docs': topic_top_docs[i]}
#          for i in range(n_topics)]
