"""
Module for exporting spaCy objects into "third-party" formats.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import operator

import spacy


def docs_to_gensim(
    spacy_docs,
    spacy_vocab,
    lemmatize=True,
    lowercase=False,
    filter_stops=True,
    filter_punct=True,
    filter_nums=False,
):
    """
    Convert a sequence of ``Doc`` s into a gensim-friendly corpus and a
    string that can be loaded into a :class:`gensim.corpora.Dictionary`.

    Args:
        spacy_docs (Iterable[:class:`spacy.tokens.Doc`])
        spacy_vocab (``spacy.Vocab``)
        lemmatize (bool): if True, use lemmatized strings for words
        lowercase (bool): if True (and ``lemmatize`` is False), use lowercased
            strings for words
        filter_stops (bool): if True, remove stop words from word list
        filter_punct (bool): if True, remove punctuation from word list
        filter_nums (bool): if True, remove numbers from word list

    Returns:
        str: words, their integer ids, and their document frequencies in
        ``spacy_docs``, as a string formatted like `id[TAB]word[TAB]df[NEWLINE]`;
        when written to file, can be converted into a gensim ``Dictionary``
        via :meth:`gensim.corpora.Dictionary.load_from_text()`

        List[List[Tuple[int, int]]]: list of documents as bags-of-words, where
        each doc is a list of (integer word ID, word count) 2-tuples
    """
    count_by = (
        spacy.attrs.LEMMA
        if lemmatize is True
        else spacy.attrs.LOWER
        if lowercase is True
        else spacy.attrs.ORTH
    )
    gcorpus = []
    stringstore = spacy.strings.StringStore()
    doc_freqs = collections.Counter()

    for spacy_doc in spacy_docs:
        bow = (
            (spacy_vocab[tok_id], count)
            for tok_id, count in spacy_doc.count_by(count_by).items()
        )
        bow = ((lex, count) for lex, count in bow if not lex.is_space)
        if filter_stops is True:
            bow = ((lex, count) for lex, count in bow if not lex.is_stop)
        if filter_punct is True:
            bow = ((lex, count) for lex, count in bow if not lex.is_punct)
        if filter_nums is True:
            bow = ((lex, count) for lex, count in bow if not lex.like_num)
        bow = sorted(
            ((stringstore[lex.orth_], count) for lex, count in bow),
            key=operator.itemgetter(0),
        )

        doc_freqs.update(tok_id for tok_id, _ in bow)
        gcorpus.append(bow)

    gdict_str = "\n".join(
        "{}\t{}\t{}".format(i, s, doc_freqs[i])
        for i, s in sorted(enumerate(stringstore), key=operator.itemgetter(1))
    )

    return (gdict_str, gcorpus)


def doc_to_conll(doc):
    """
    Convert a single ``Doc`` into a CoNLL-U formatted str, as described at
    http://universaldependencies.org/docs/format.html.

    Args:
        doc (:class:`spacy.tokens.Doc`)

    Returns:
        str

    Raises:
        ValueError: if ``doc`` is not parsed
    """
    if doc.is_parsed is False:
        raise ValueError("spaCy doc must be parsed")
    rows = []
    for j, sent in enumerate(doc.sents):
        sent_i = sent.start
        sent_id = j + 1
        rows.append("# sent_id {}".format(sent_id))
        for i, tok in enumerate(sent):
            # HACK...
            if tok.is_space:
                form = " "
                lemma = " "
            else:
                form = tok.orth_
                lemma = tok.lemma_
            tok_id = i + 1
            head = tok.head.i - sent_i + 1
            if head == tok_id:
                head = 0
            misc = "SpaceAfter=No" if not tok.whitespace_ else "_"
            rows.append(
                "\t".join(
                    [
                        str(tok_id),
                        form,
                        lemma,
                        tok.pos_,
                        tok.tag_,
                        "_",
                        str(head),
                        tok.dep_.lower(),
                        "_",
                        misc,
                    ]
                )
            )
        rows.append("")  # sentences must be separated by a single newline
    return "\n".join(rows)
