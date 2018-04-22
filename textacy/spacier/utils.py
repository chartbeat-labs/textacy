"""
spaCy Utils
-----------

WIP
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from spacy import attrs
from spacy.language import Language as SpacyLang

from .. import cache
from .. import compat


def make_doc_from_text_chunks(text, lang, chunk_size=100000):
    """
    Make a single spaCy-processed document from 1 or more chunks of ``text``.
    This is a workaround for processing very long texts, for which spaCy
    is unable to allocate enough RAM.

    Although this function's performance is *pretty good*, it's inherently
    less performant that just processing the entire text in one shot.
    Only use it if necessary!

    Args:
        text (str): Text document to be chunked and processed by spaCy.
        lang (str or ``spacy.Language``): A 2-letter language code (e.g. "en"),
            the name of a spaCy model for the desired language, or
            an already-instantiated spaCy language pipeline.
        chunk_size (int): Number of characters comprising each text chunk
            (excluding the last chunk, which is probably smaller). For best
            performance, value should be somewhere between 1e3 and 1e7,
            depending on how much RAM you have available.

            .. note:: Since chunking is done by character, chunks edges' probably
               won't respect natural language segmentation, which means that every
               ``chunk_size`` characters, spaCy will probably get tripped up and
               make weird parsing errors.

    Returns:
        ``spacy.Doc``: A single processed document, initialized from
        components accumulated chunk by chunk.
    """
    if isinstance(lang, compat.unicode_):
        spacy_lang = cache.load_spacy(lang)
    elif not isinstance(lang, SpacyLang):
        raise TypeError(
            '`lang` must be {}, not {}'.format({compat.unicode_, SpacyLang}, type(lang)))

    words = []
    spaces = []
    np_arrays = []
    cols = [attrs.POS, attrs.TAG, attrs.DEP, attrs.HEAD, attrs.ENT_IOB, attrs.ENT_TYPE]
    text_len = len(text)
    i = 0
    # iterate over text chunks and accumulate components needed to make a doc
    while i < text_len:
        chunk_doc = spacy_lang(text[i: i + chunk_size])
        words.extend(tok.text for tok in chunk_doc)
        spaces.extend(bool(tok.whitespace_) for tok in chunk_doc)
        np_arrays.append(chunk_doc.to_array(cols))
        i += chunk_size
    # now, initialize the doc from words and spaces
    # then load attribute values from the concatenated np array
    doc = spacy.tokens.Doc(spacy_lang.vocab, words=words, spaces=spaces)
    doc = doc.from_array(cols, np.concatenate(np_arrays, axis=0))

    return doc
