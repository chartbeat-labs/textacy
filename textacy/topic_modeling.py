"""
Module to facilitate topic modeling with gensim. For example::

    >>> texts = (text for _, _, text in
    ...          textacy.corpora.wikipedia.get_plaintext_pages('enwiki-latest-pages-articles.xml.bz2', max_n_pages=1000))
    >>> proc_texts = (proc_text for proc_text in preprocess_texts(texts))
    >>> spacy_docs = (spacy_doc for spacy_doc in
    ...               texts_to_spacy_docs(proc_texts, merge_nes=True, merge_nps=False))
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
import gensim
from gensim.corpora.dictionary import Dictionary as GensimDictionary
import numpy as np
from sklearn.decomposition import NMF

from textacy import data, extract, fileio, preprocess, spacy_utils

# TODO: what to do about lang?

def preprocess_texts(texts):
    """
    Default preprocessing for raw texts before topic modeling.

    Args:
        texts (iterable(str))

    Yields:
        str
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
        lang (str, {'en'})
        merge_nes (bool, optional)
        merge_ncs (bool, optional)

    Yields:
        ``spacy.Doc``
    """
    spacy_nlp = data.load_spacy_pipeline(
        lang=lang, entity=merge_nes, parser=merge_nps)
    for spacy_doc in spacy_nlp.pipe(texts, tag=True, parse=merge_nps, entity=merge_nes,
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
        return (gdict, [gdict.doc2bow(term_list, allow_update=False) for term_list in term_lists])
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


def _train_lda(corpus, dictionary, n_topics, n_passes, save=False):
    """
    """
    lda = gensim.models.LdaModel(corpus, id2word=dictionary,
                                 num_topics=n_topics, passes=n_passes)
    if save:
        # TODO: make this py2-3 compatible a la
        # https://github.com/piskvorky/gensim/wiki/Recipes-&-FAQ#q9-how-do-i-load-a-model-in-python-3-that-was-trained-and-saved-using-python-2
        lda.save(save)
    # get top N terms per topic

    # transform documents into topic representation
    # normalize such that sum of topic contributions per doc = 1
    # ^ this is approximately true already, except gensim truncates low-weighted topics
    # resulting in sum(topic_weights) ~ 0.95
    doc_topics = np.array([gensim.matutils.sparse2full(lda[doc], n_topics) for doc in corpus])
    doc_topics = doc_topics / np.sum(doc_topics, axis=1, keepdims=True)


def get_top_topic_terms(model, n_terms=20, n_topics=-1):
    """
    """
    return [[term for term, _ in topic]
            for _, topic in
            model.show_topics(num_topics=n_topics, num_words=n_top_terms,
                              log=False, formatted=False)
            ]


def get_top_topic_docs(model, corpus, n_topics=-1):
    """
    """
    doc_topics = np.array([gensim.matutils.sparse2full(model[doc], n_topics) for doc in corpus])
    doc_topics = doc_topics / np.sum(doc_topics, axis=1, keepdims=True)
    topic_top_docs = [list(np.argsort(doc_topics[:, i])[::-1][:n_top_docs])
                      for i in range(n_topics)]


def train_topic_model(corpus, dictionary, algorithm, n_topics,
                      n_top_terms=25, n_top_docs=10,
                      save_to_disk=None,
                      **kwargs):
    """
    Train a topic model (NMF, LDA, or HDP) on a corpus, return a summary of the
    model as JSON.

    Args:
        corpus (list of list of 2-tuples)
        dictionary (id:term mapping):
            e.g. gensim.corpora.dictionary.Dictionary instance
        model (str, {'nmf', 'lda', 'hdp'}):
            topic model(s) to train/save
        n_topics (int, optional): number of topics in the model
        n_top_terms (int, optional): number of top-weighted terms to save with each topic
        n_top_docs (int, optional): number of top-weighted documents to save with each topic
        save_to_disk (str, optional):
            gives /path/to/fname where topic model can be saved to disk
        **kwargs :
            max_nmf_iter : int, optional
            n_lda_passes : int, optional
            hdp_K : int, optional

    Returns:
        list(dict):
            each dict corresponds to one topic in the model
            with keys `topic_id`, `topic_weight`, `key_terms`, `key_docs`
    """
    import joblib
    fname = model + '-' + str(n_topics) + '-topics'
    if model == 'nmf':
        # train model
        csr_corpus = gensim.matutils.corpus2csc(corpus).transpose()
        nmf = NMF(n_components=n_topics,
                  random_state=1,
                  max_iter=kwargs.get('max_nmf_iter', 400)
                  ).fit(csr_corpus)
        if save_to_disk is not None:
            joblib.dump(nmf, save_to_disk, compress=3)
        # get top N terms per topic
        topic_top_terms = [[dictionary[i] for i in np.argsort(topic)[::-1][:n_top_terms]]
                           for topic in nmf.components_]
        # transform documents into topic representation
        # normalize such that sum of topic contributions per doc = 1
        doc_topics = nmf.transform(csr_corpus)
        doc_topics = doc_topics / np.sum(doc_topics, axis=1, keepdims=True)

    elif model == 'lda':
        # train model
        lda = gensim.models.LdaModel(corpus,
                                     num_topics=n_topics,
                                     id2word=dictionary,
                                     passes=kwargs.get('n_lda_passes', 10))
        if save_to_disk is not None:
            lda.save(save_to_disk)
        # get top N terms per topic
        topic_top_terms = [[sw[0] for sw in topic]
            for topic_idx, topic in lda.show_topics(num_topics=-1, num_words=n_top_terms, log=False, formatted=False)]
        # transform documents into topic representation
        # normalize such that sum of topic contributions per doc = 1
        # ^ this is approximately true already, except gensim truncates low-weighted topics
        # resulting in sum(topic_weights) ~ 0.95
        doc_topics = np.array([gensim.matutils.sparse2full(lda[doc], n_topics) for doc in corpus])
        doc_topics = doc_topics / np.sum(doc_topics, axis=1, keepdims=True)

    elif model == 'hdp':
        # train model
        hdp = gensim.models.HdpModel(corpus, dictionary,
                                     T=n_topics, K=kwargs.get('hdp_K', 5),
                                     var_converge=0.000000001,
                                     kappa=0.8)
        hdp.optimal_ordering()
        if save_to_disk is not None:
            hdp.save(save_to_disk)
        # get top N terms per topic
        topic_top_terms = [[sw[0] for sw in topic[1]]
            for topic in hdp.show_topics(topics=-1, topn=n_top_terms, log=False, formatted=False)]
        # transform documents into topic representation
        # normalize such that sum of topic contributions per doc = 1
        # ^ this is approximately true already, except gensim truncates low-weighted topics
        # resulting in sum(topic_weights) ~ 0.95
        doc_topics = np.array([gensim.matutils.sparse2full(hdp[doc], n_topics) for doc in corpus])
        doc_topics = doc_topics / np.sum(doc_topics, axis=1, keepdims=True)

    # get *normalized* topic weights in corpus
    topic_weights = [sum(doc_topics[:,i]) for i in range(n_topics)]
    topic_weights = [tw / sum(topic_weights) for tw in topic_weights]
    # get top documents per topic
    topic_top_docs = [list(np.argsort(doc_topics[:,i])[::-1][:n_top_docs])
                      for i in range(n_topics)]

    return [{'topic_id': i,
             'topic_weight': topic_weights[i],
             'key_terms': topic_top_terms[i],
             'key_docs': topic_top_docs[i]}
             for i in range(n_topics)]
