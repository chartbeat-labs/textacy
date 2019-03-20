# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

import pytest

from textacy import Doc
from textacy import cache
from textacy import compat

TEXT = """
Since the so-called "statistical revolution" in the late 1980s and mid 1990s, much Natural Language Processing research has relied heavily on machine learning.
Formerly, many language-processing tasks typically involved the direct hand coding of rules, which is not in general robust to natural language variation. The machine-learning paradigm calls instead for using statistical inference to automatically learn such rules through the analysis of large corpora of typical real-world examples (a corpus is a set of documents, possibly with human or computer annotations).
Many different classes of machine learning algorithms have been applied to NLP tasks. These algorithms take as input a large set of "features" that are generated from the input data. Some of the earliest-used algorithms, such as decision trees, produced systems of hard if-then rules similar to the systems of hand-written rules that were then common. Increasingly, however, research has focused on statistical models, which make soft, probabilistic decisions based on attaching real-valued weights to each input feature. Such models have the advantage that they can express the relative certainty of many different possible answers rather than only one, producing more reliable results when such a model is included as a component of a larger system.
"""


@pytest.fixture(scope="module")
def doc(request):
    return Doc(TEXT.strip(), lang="en", metadata={"foo": "bar!"})


# init tests


def test_unicode_content():
    assert isinstance(Doc("This is an English sentence."), Doc)


def test_spacydoc_content():
    spacy_lang = cache.load_spacy("en")
    spacy_doc = spacy_lang("This is an English sentence.")
    assert isinstance(Doc(spacy_doc), Doc)


def test_invalid_content():
    invalid_contents = [
        b"This is an English sentence in bytes.",
        {"content": "This is an English sentence as dict value."},
        True,
    ]
    for invalid_content in invalid_contents:
        with pytest.raises(ValueError):
            _ = Doc(invalid_content)


def test_lang_str():
    assert isinstance(Doc("This is an English sentence.", lang="en"), Doc)


def test_lang_spacylang():
    spacy_lang = cache.load_spacy("en")
    assert isinstance(Doc("This is an English sentence.", lang=spacy_lang), Doc)


def test_lang_callable():
    def dumb_detect_language(text):
        return "en"

    assert isinstance(
        Doc("This is an English sentence.", lang=dumb_detect_language), Doc
    )
    assert isinstance(Doc("This is an English sentence.", lang=lambda x: "en"), Doc)


def test_invalid_lang():
    invalid_langs = [b"en", ["en", "en_core_web_sm"], True]
    for invalid_lang in invalid_langs:
        with pytest.raises(TypeError):
            _ = Doc("This is an English sentence.", lang=invalid_lang)


def test_invalid_content_lang_combo():
    spacy_lang = cache.load_spacy("en")
    with pytest.raises(ValueError):
        _ = Doc(spacy_lang("Hola, cómo estás mi amigo?"), lang="es")


# methods tests


def test_n_tokens_and_sents(doc):
    n_tokens = doc.n_tokens
    n_sents = doc.n_sents
    assert isinstance(n_sents, int) and n_sents > 0
    assert isinstance(n_tokens, int) and n_tokens > 0


def test_term_count(doc):
    count1 = doc.count("statistical")
    count2 = doc.count("machine learning")
    count3 = doc.count("foo")
    assert isinstance(count1, int) and count1 > 0
    assert isinstance(count2, int) and count2 > 0
    assert isinstance(count3, int) and count3 == 0


def test_tokenized_text(doc):
    tokenized_text = doc.tokenized_text
    assert isinstance(tokenized_text, list)
    assert isinstance(tokenized_text[0], list)
    assert isinstance(tokenized_text[0][0], compat.unicode_)
    assert len(tokenized_text) == doc.n_sents


def test_pos_tagged_text(doc):
    pos_tagged_text = doc.pos_tagged_text
    assert isinstance(pos_tagged_text, list)
    assert isinstance(pos_tagged_text[0], list)
    assert isinstance(pos_tagged_text[0][0], tuple)
    assert isinstance(pos_tagged_text[0][0][0], compat.unicode_)
    assert len(pos_tagged_text) == doc.n_sents


def test_to_terms_list(doc):
    full_terms_list = list(doc.to_terms_list(as_strings=True))
    full_terms_list_ids = list(doc.to_terms_list(as_strings=False))
    assert len(full_terms_list) == len(full_terms_list_ids)
    assert isinstance(full_terms_list[0], compat.unicode_)
    assert isinstance(full_terms_list_ids[0], compat.int_types)
    assert (
        full_terms_list[0]
        != list(doc.to_terms_list(as_strings=True, normalize=False))[0]
    )
    assert len(list(doc.to_terms_list(ngrams=False))) < len(full_terms_list)
    assert len(list(doc.to_terms_list(ngrams=1))) < len(full_terms_list)
    assert len(list(doc.to_terms_list(ngrams=(1, 2)))) < len(full_terms_list)
    assert len(list(doc.to_terms_list(ngrams=False))) < len(full_terms_list)


def test_to_bag_of_words(doc):
    bow = doc.to_bag_of_words(weighting="count")
    assert isinstance(bow, dict)
    assert isinstance(list(bow.keys())[0], compat.int_types)
    assert isinstance(list(bow.values())[0], int)
    bow = doc.to_bag_of_words(weighting="binary")
    assert isinstance(bow, dict)
    assert isinstance(list(bow.keys())[0], compat.int_types)
    assert isinstance(list(bow.values())[0], int)
    for value in list(bow.values())[0:10]:
        assert value < 2
    bow = doc.to_bag_of_words(weighting="freq")
    assert isinstance(bow, dict)
    assert isinstance(list(bow.keys())[0], compat.int_types)
    assert isinstance(list(bow.values())[0], float)
    bow = doc.to_bag_of_words(as_strings=True)
    assert isinstance(bow, dict)
    assert isinstance(list(bow.keys())[0], compat.unicode_)


def test_doc_save_and_load(tmpdir, doc):
    filepath = str(tmpdir.join("test_doc_save_and_load.pkl"))
    doc.save(filepath)
    new_doc = Doc.load(filepath)
    assert isinstance(new_doc, Doc)
    assert len(new_doc) == len(doc)
    assert new_doc.lang == doc.lang
    assert new_doc.metadata == doc.metadata


def test_to_semantic_network_words(doc):
    graph = doc.to_semantic_network(nodes="words", edge_weighting="cooc_freq")
    assert all(isinstance(node, compat.unicode_) for node in graph.nodes)
    assert all(isinstance(d["weight"], int) for n1, n2, d in graph.edges(data=True))
    graph = doc.to_semantic_network(nodes="words", edge_weighting="binary")
    assert all(isinstance(node, compat.unicode_) for node in graph.nodes)
    assert all(d == {} for n1, n2, d in graph.edges(data=True))


def test_to_semantic_network_sents(doc):
    graph = doc.to_semantic_network(nodes="sents", edge_weighting="cosine")
    assert all(isinstance(node, int) for node in graph.nodes)
    assert all(isinstance(d["weight"], float) for n1, n2, d in graph.edges(data=True))
    graph = doc.to_semantic_network(nodes="sents", edge_weighting="jaccard")
    assert all(isinstance(node, int) for node in graph.nodes)
    assert all(isinstance(d["weight"], int) for n1, n2, d in graph.edges(data=True))
