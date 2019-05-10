# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

import pytest
import spacy

from textacy import cache
from textacy import compat
from textacy.doc import (
    make_spacy_doc,
    remove_doc_extensions,
    set_doc_extensions,
    _doc_extensions,
)

TEXT = """
Since the so-called "statistical revolution" in the late 1980s and mid 1990s, much Natural Language Processing research has relied heavily on machine learning.
Formerly, many language-processing tasks typically involved the direct hand coding of rules, which is not in general robust to natural language variation. The machine-learning paradigm calls instead for using statistical inference to automatically learn such rules through the analysis of large corpora of typical real-world examples (a corpus is a set of documents, possibly with human or computer annotations).
Many different classes of machine learning algorithms have been applied to NLP tasks. These algorithms take as input a large set of "features" that are generated from the input data. Some of the earliest-used algorithms, such as decision trees, produced systems of hard if-then rules similar to the systems of hand-written rules that were then common. Increasingly, however, research has focused on statistical models, which make soft, probabilistic decisions based on attaching real-valued weights to each input feature. Such models have the advantage that they can express the relative certainty of many different possible answers rather than only one, producing more reliable results when such a model is included as a component of a larger system.
""".strip()


@pytest.fixture(scope="module")
def doc(request):
    return make_spacy_doc((TEXT, {"foo": "bar!"}), lang="en")


class TestMakeSpacyDoc(object):

    def test_unicode_content(self):
        assert isinstance(
            make_spacy_doc("This is an English sentence."),
            spacy.tokens.Doc,
        )

    def test_spacydoc_content(self):
        spacy_lang = cache.load_spacy("en")
        spacy_doc = spacy_lang("This is an English sentence.")
        assert isinstance(
            make_spacy_doc(spacy_doc),
            spacy.tokens.Doc,
        )

    def test_invalid_content(self):
        invalid_contents = [
            b"This is an English sentence in bytes.",
            {"content": "This is an English sentence as dict value."},
            True,
        ]
        for invalid_content in invalid_contents:
            with pytest.raises(TypeError):
                _ = make_spacy_doc(invalid_content)

    def test_lang_str(self):
        assert isinstance(
            make_spacy_doc("This is an English sentence.", lang="en"),
            spacy.tokens.Doc,
        )

    def test_lang_spacylang(self):
        spacy_lang = cache.load_spacy("en")
        assert isinstance(
            make_spacy_doc("This is an English sentence.", lang=spacy_lang),
            spacy.tokens.Doc,
        )

    def test_lang_callable(self):
        def dumb_detect_language(text):
            return "en"

        assert isinstance(
            make_spacy_doc("This is an English sentence.", lang=dumb_detect_language),
            spacy.tokens.Doc,
        )
        assert isinstance(
            make_spacy_doc("This is an English sentence.", lang=lambda x: "en"),
            spacy.tokens.Doc,
        )

    def test_invalid_lang(self):
        invalid_langs = [b"en", ["en", "en_core_web_sm"], True]
        for invalid_lang in invalid_langs:
            with pytest.raises(TypeError):
                _ = make_spacy_doc("This is an English sentence.", lang=invalid_lang)

    def test_invalid_content_lang_combo(self):
        spacy_lang = cache.load_spacy("en")
        with pytest.raises(ValueError):
            _ = make_spacy_doc(spacy_lang("Hola, cómo estás mi amigo?"), lang="es")


def test_set_remove_extensions():
    remove_doc_extensions()
    for name in _doc_extensions.keys():
        assert spacy.tokens.Doc.has_extension(name) is False
    set_doc_extensions()
    for name in _doc_extensions.keys():
        assert spacy.tokens.Doc.has_extension(name) is True


class TestDocExtensions(object):

    def test_n_tokens(self, doc):
        n_tokens = doc._.n_tokens
        assert isinstance(n_tokens, int) and n_tokens > 0

    def test_n_sents(self, doc):
        n_sents = doc._.n_sents
        assert isinstance(n_sents, int) and n_sents > 0

    # TODO: re-add this test if count() gets implemented
    # def test_term_count(self, doc):
    #     count1 = doc.count("statistical")
    #     count2 = doc.count("machine learning")
    #     count3 = doc.count("foo")
    #     assert isinstance(count1, int) and count1 > 0
    #     assert isinstance(count2, int) and count2 > 0
    #     assert isinstance(count3, int) and count3 == 0

    def test_to_tokenized_text(self, doc):
        tokenized_text = doc._.to_tokenized_text()
        assert isinstance(tokenized_text, list)
        assert isinstance(tokenized_text[0], list)
        assert isinstance(tokenized_text[0][0], compat.unicode_)
        assert len(tokenized_text) == doc._.n_sents

    def test_to_tagged_text(self, doc):
        tagged_text = doc._.to_tagged_text()
        assert isinstance(tagged_text, list)
        assert isinstance(tagged_text[0], list)
        assert isinstance(tagged_text[0][0], tuple)
        assert isinstance(tagged_text[0][0][0], compat.unicode_)
        assert len(tagged_text) == doc._.n_sents

    def test_to_terms_list(self, doc):
        full_terms_list = list(doc._.to_terms_list(as_strings=True))
        full_terms_list_ids = list(doc._.to_terms_list(as_strings=False))
        assert len(full_terms_list) == len(full_terms_list_ids)
        assert isinstance(full_terms_list[0], compat.unicode_)
        assert isinstance(full_terms_list_ids[0], compat.int_types)
        assert (
            full_terms_list[0]
            != list(doc._.to_terms_list(as_strings=True, normalize=False))[0]
        )
        assert len(list(doc._.to_terms_list(ngrams=False))) < len(full_terms_list)
        assert len(list(doc._.to_terms_list(ngrams=1))) < len(full_terms_list)
        assert len(list(doc._.to_terms_list(ngrams=(1, 2)))) < len(full_terms_list)
        assert len(list(doc._.to_terms_list(ngrams=False))) < len(full_terms_list)

    def test_to_bag_of_words(self, doc):
        bow = doc._.to_bag_of_words(weighting="count")
        assert isinstance(bow, dict)
        assert isinstance(list(bow.keys())[0], compat.int_types)
        assert isinstance(list(bow.values())[0], int)
        bow = doc._.to_bag_of_words(weighting="binary")
        assert isinstance(bow, dict)
        assert isinstance(list(bow.keys())[0], compat.int_types)
        assert isinstance(list(bow.values())[0], int)
        for value in list(bow.values())[0:10]:
            assert value < 2
        bow = doc._.to_bag_of_words(weighting="freq")
        assert isinstance(bow, dict)
        assert isinstance(list(bow.keys())[0], compat.int_types)
        assert isinstance(list(bow.values())[0], float)
        bow = doc._.to_bag_of_words(as_strings=True)
        assert isinstance(bow, dict)
        assert isinstance(list(bow.keys())[0], compat.unicode_)

    # def test_doc_save_and_load(self, tmpdir, doc):
    #     filepath = str(tmpdir.join("test_doc_save_and_load.pkl"))
    #     doc.save(filepath)
    #     new_doc = Doc.load(filepath)
    #     assert isinstance(new_doc, Doc)
    #     assert len(new_doc) == len(doc)
    #     assert new_doc.lang == doc.lang
    #     assert new_doc.metadata == doc.metadata

    def test_to_semantic_network_words(self, doc):
        graph = doc._.to_semantic_network(nodes="words", edge_weighting="cooc_freq")
        assert all(isinstance(node, compat.unicode_) for node in graph.nodes)
        assert all(isinstance(d["weight"], int) for n1, n2, d in graph.edges(data=True))
        graph = doc._.to_semantic_network(nodes="words", edge_weighting="binary")
        assert all(isinstance(node, compat.unicode_) for node in graph.nodes)
        assert all(d == {} for n1, n2, d in graph.edges(data=True))

    def test_to_semantic_network_sents(self, doc):
        graph = doc._.to_semantic_network(nodes="sents", edge_weighting="cosine")
        assert all(isinstance(node, int) for node in graph.nodes)
        assert all(isinstance(d["weight"], float) for n1, n2, d in graph.edges(data=True))
        graph = doc._.to_semantic_network(nodes="sents", edge_weighting="jaccard")
        assert all(isinstance(node, int) for node in graph.nodes)
        assert all(isinstance(d["weight"], int) for n1, n2, d in graph.edges(data=True))
