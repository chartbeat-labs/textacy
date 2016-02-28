"""
Module to facilitate topic modeling with gensim. For example::

    >>> filename = '/Users/burtondewilde/Desktop/wiki_spacy_docs_v2.bin'
    >>> texts = (text for _, _, text in wikipedia.get_plaintext_pages(WIKI_FILE, max_n_pages=1000))
    >>> preprocessed_texts = (ptext for ptext in preprocess_texts(texts))
    >>> spacy_docs = (spacy_doc for spacy_doc in texts_to_spacy_docs(preprocessed_texts, merge_nes=True, merge_nps=False))
    >>> fileio.write_spacy_docs(spacy_docs, filename)
    >>> spacy_docs = (spacy_doc for spacy_doc in fileio.read_spacy_docs(data.load_spacy_pipeline().vocab, filename))
    >>> term_lists = (term_list for term_list in
                      spacy_docs_to_term_lists(spacy_docs, lemmatize=True,
                                               filter_stops=True, filter_punct=True, filter_nums=False,
                                               good_pos_tags=None, bad_pos_tags=None))
    >>> gdict, gcorpus = term_lists_to_gensim_corpus(
            list(term_lists), min_doc_count=1, max_doc_freq=1.0, keep_top_n=False)
    >>> topics = train_topic_model(gcorpus, gdict, 'lda', 10,
                                   n_top_terms=25, n_top_docs=10,
                                   save_to_disk=None)
"""
import gensim
from gensim.corpora.dictionary import Dictionary as GensimDictionary
import numpy as np
from sklearn.decomposition import NMF

from textacy import data, extract, fileio, preprocess, spacy_utils

# TODO: what to do about lang?

def preprocess_texts(texts):
    for text in texts:
        yield preprocess.preprocess_text(
            text, no_urls=True, no_emails=True, no_phone_numbers=True)


def texts_to_spacy_docs(texts, lang, merge_nes=True, merge_nps=False):
    spacy_nlp = data.load_spacy_pipeline(lang=lang,
                                         entity=merge_nes, parser=merge_nps)
    for text in texts:
        spacy_doc = spacy_nlp(text)
        if merge_nes is True:
            spacy_utils.merge_spans(
                extract.named_entities(
                spacy_doc, bad_ne_types='numeric', drop_determiners=False))
        if merge_nps is True:
            spacy_utils.merge_spans(
                extract.noun_chunks(
                    spacy_doc, drop_determiners=True))
        yield spacy_doc

# optional: save spacy docs to disk!
# >>> fileio.write_spacy_docs(texts_to_spacy_docs(texts), <filename>)

def spacy_docs_to_term_lists(spacy_docs, lemmatize=True,
                             filter_stops=True, filter_punct=True, filter_nums=False,
                             good_pos_tags=None, bad_pos_tags=None):
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


def term_lists_to_gensim_corpus(term_lists, min_doc_count=1, max_doc_freq=1.0, keep_top_n=False):
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
        return (gdict, [gdict.doc2bow(term_list, allow_update=True) for term_list in term_lists])


def weight_gensim_corpus(corpus, weighting='tfidf'):
    if weighting == 'tfidf':
        tfidf_model = gensim.models.TfidfModel(corpus, normalize=True)
        corpus = tfidf_model[corpus]
    elif weighting == 'logent':
        logent_model = gensim.models.LogEntropyModel(corpus, normalize=True)
        corpus = logent_model[corpus]
    return corpus


def train_topic_model(corpus, dictionary,
                      model, n_topics,
                      n_top_terms=25,
                      n_top_docs=10,
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
