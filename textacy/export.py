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


def doc_to_conll(doc, save_to=False):
    """
    Convert a single ``spacy.Doc`` into CoNLL-U string format, optionally saving to disk.

    Args:
        doc (``spacy.Doc``)
        save_to (str, optional): to save the CoNLL string to disk, provide the full
            path/to/fname.txt; otherwise, the string is returned but not saved

    Returns:
        str or None
    """
    rows = []
    for j, sent in enumerate(doc.sents):
        sent_i = sent.start
        sent_id = j + 1
        rows.append('# sent_id {}'.format(sent_id))
        for i, tok in enumerate(sent):
            # HACK...
            if tok.is_space:
                form = ' '
                lemma = ' '
            else:
                form = tok.orth_
                lemma = tok.lemma_
            tok_id = i + 1
            head = tok.head.i - sent_i + 1
            if head == tok_id:
                head = 0
            misc = 'SpaceAfter=No' if not tok.whitespace_ else '_'
            rows.append('\t'.join([str(tok_id), form, lemma, tok.pos_, tok.tag_,
                                   '_', str(head), tok.dep_.lower(), '_', misc]))
        rows.append('')  # sentences must be separated by a single newline
    conll = '\n'.join(rows)
    if save_to is False:
        return conll
    else:
        with io.open(save_to, mode='w', encoding='utf-8') as f:
            f.write(conll)
