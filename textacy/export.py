"""
Module for exporting textacy/spacy objects into "third-party" formats.
"""
import io

from gensim.corpora.dictionary import Dictionary

from textacy import extract


def doc_to_gensim(doc, lemmatize=True,
                  filter_stops=True, filter_punct=True, filter_nums=False):
    """
    Convert a single ``spacy.Doc`` into a gensim dictionary and bag-of-words document.

    Args:
        doc (``spacy.Doc``)
        lemmatize (bool): if True, use lemmatized strings for words; otherwise,
            use the original form of the string as it appears in ``doc``
        filter_stops (bool): if True, remove stop words from word list
        filter_punct (bool): if True, remove punctuation from word list
        filter_nums (bool): if True, remove numbers from word list

    Returns:
        :class:`gensim.Dictionary <gensim.corpora.dictionary.Dictionary>`:
            integer word ID to word string mapping
        list((int, int)): bag-of-words document, a list of (integer word ID, word count)
            2-tuples
    """
    gdict = Dictionary()
    words = extract.words(doc,
                          filter_stops=filter_stops,
                          filter_punct=filter_punct,
                          filter_nums=filter_nums)
    if lemmatize is True:
        gdoc = gdict.doc2bow((word.lemma_ for word in words), allow_update=True)
    else:
        gdoc = gdict.doc2bow((word.orth_ for word in words), allow_update=True)

    return (gdict, gdoc)


def docs_to_gensim(docs, lemmatize=True,
                   filter_stops=True, filter_punct=True, filter_nums=False):
    """
    Convert multiple ``spacy.Doc`` s into a gensim dictionary and bag-of-words corpus.

    Args:
        docs (list(``spacy.Doc``))
        lemmatize (bool): if True, use lemmatized strings for words; otherwise,
            use the original form of the string as it appears in ``doc``
        filter_stops (bool): if True, remove stop words from word list
        filter_punct (bool): if True, remove punctuation from word list
        filter_nums (bool): if True, remove numbers from word list

    Returns:
        :class:`gensim.Dictionary <gensim.corpora.dictionary.Dictionary>`:
            integer word ID to word string mapping
        list(list((int, int))): list of bag-of-words documents, where each doc is
            a list of (integer word ID, word count) 2-tuples
    """
    gdict = Dictionary()
    gcorpus = []
    for doc in docs:
        words = extract.words(doc,
                              filter_stops=filter_stops,
                              filter_punct=filter_punct,
                              filter_nums=filter_nums)
        if lemmatize is True:
            gcorpus.append(gdict.doc2bow((word.lemma_ for word in words), allow_update=True))
        else:
            gcorpus.append(gdict.doc2bow((word.orth_ for word in words), allow_update=True))

    return (gdict, gcorpus)
