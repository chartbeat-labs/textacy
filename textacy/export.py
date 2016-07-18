"""
Module for exporting textacy/spacy objects into "third-party" formats.
"""
from collections import Counter
from operator import itemgetter

from gensim.corpora.dictionary import Dictionary
from spacy import attrs
from spacy.strings import StringStore

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


def docs_to_gensim(spacy_docs, spacy_vocab, lemmatize=True,
                   filter_stops=True, filter_punct=True, filter_nums=False):
    """
    Convert multiple ``spacy.Doc`` s into a gensim dictionary and bag-of-words corpus.

    Args:
        spacy_docs (list(``spacy.Doc``))
        spacy_vocab (``spacy.Vocab``)
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
    stringstore = StringStore()
    doc_freqs = Counter()

    for spacy_doc in spacy_docs:
        if lemmatize is True:
            bow = ((spacy_vocab[tok_id], count)
                   for tok_id, count in spacy_doc.count_by(attrs.LEMMA).items())
        else:
            bow = ((spacy_vocab[tok_id], count)
                   for tok_id, count in spacy_doc.count_by(attrs.ORTH).items())

        if filter_stops is True:
            bow = ((lex, count) for lex, count in bow if not lex.is_stop)
        if filter_punct is True:
            bow = ((lex, count) for lex, count in bow if not lex.is_punct)
        if filter_nums is True:
            bow = ((lex, count) for lex, count in bow if not lex.like_num)

        bow = sorted(((stringstore[lex.orth_], count) for lex, count in bow),
                     key=itemgetter(0))

        doc_freqs.update(tok_id for tok_id, _ in bow)
        gdict.num_docs += 1
        gdict.num_pos += sum(count for _, count in bow)
        gdict.num_nnz += len(bow)

        gcorpus.append(bow)

    gdict.token2id = {s: i for i, s in enumerate(stringstore)}
    gdict.dfs = dict(doc_freqs)

    return (gdict, gcorpus)


def doc_to_conll(doc):
    """
    Convert a single ``spacy.Doc`` into a CoNLL-U formatted str, as described at
    http://universaldependencies.org/docs/format.html.

    Args:
        doc (``spacy.Doc``)

    Returns:
        str

    Raises:
        ValueError: if ``doc`` is not parsed
    """
    if doc.is_parsed is False:
        raise ValueError('spaCy doc must be parsed')
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
    return '\n'.join(rows)
