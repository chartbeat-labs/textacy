import pytest
import spacy

from textacy import load_spacy_lang, make_spacy_doc
from textacy.spacier.doc_extensions import (
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
    return make_spacy_doc((TEXT, {"foo": "bar"}), lang="en_core_web_sm")


def test_set_remove_extensions():
    remove_doc_extensions()
    for name in _doc_extensions.keys():
        assert spacy.tokens.Doc.has_extension(name) is False
    set_doc_extensions()
    for name in _doc_extensions.keys():
        assert spacy.tokens.Doc.has_extension(name) is True


class TestDocExtensions:

    def test_lang(self, doc):
        lang = doc._.lang
        assert isinstance(lang, str)
        assert lang == doc.vocab.lang

    def test_preview(self, doc):
        preview = doc._.preview
        assert isinstance(preview, str)
        assert preview.startswith("Doc")

    def test_tokens(self, doc):
        tokens = list(doc._.tokens)[:5]
        assert all(isinstance(token, spacy.tokens.Token) for token in tokens)

    def test_n_tokens(self, doc):
        n_tokens = doc._.n_tokens
        assert isinstance(n_tokens, int) and n_tokens > 0

    def test_n_sents(self, doc):
        n_sents = doc._.n_sents
        assert isinstance(n_sents, int) and n_sents > 0

    def test_meta(self, doc):
        meta = doc._.meta
        assert isinstance(meta, dict)
        with pytest.raises(TypeError):
            doc._.meta = None

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
        assert isinstance(tokenized_text[0][0], str)
        assert len(tokenized_text) == doc._.n_sents

    def test_to_tokenized_text_nosents(self):
        spacy_lang = load_spacy_lang("en_core_web_sm")
        with spacy_lang.select_pipes(disable="parser"):
            doc = spacy_lang("This is sentence #1. This is sentence #2.")
        tokenized_text = doc._.to_tokenized_text()
        assert isinstance(tokenized_text, list)
        assert len(tokenized_text) == 1
        assert isinstance(tokenized_text[0], list)
        assert isinstance(tokenized_text[0][0], str)

    def test_to_tagged_text(self, doc):
        tagged_text = doc._.to_tagged_text()
        assert isinstance(tagged_text, list)
        assert isinstance(tagged_text[0], list)
        assert isinstance(tagged_text[0][0], tuple)
        assert isinstance(tagged_text[0][0][0], str)
        assert len(tagged_text) == doc._.n_sents

    def test_to_tagged_text_nosents(self):
        spacy_lang = load_spacy_lang("en_core_web_sm")
        with spacy_lang.select_pipes(disable="parser"):
            doc = spacy_lang("This is sentence #1. This is sentence #2.")
        tagged_text = doc._.to_tagged_text()
        assert isinstance(tagged_text, list)
        assert len(tagged_text) == 1
        assert isinstance(tagged_text[0], list)
        assert isinstance(tagged_text[0][0], tuple)
        assert isinstance(tagged_text[0][0][0], str)

    def test_to_terms_list(self, doc):
        full_terms_list = list(doc._.to_terms_list(as_strings=True))
        full_terms_list_ids = list(doc._.to_terms_list(as_strings=False))
        assert len(full_terms_list) == len(full_terms_list_ids)
        assert isinstance(full_terms_list[0], str)
        assert isinstance(full_terms_list_ids[0], int)
        assert (
            full_terms_list[0]
            != list(doc._.to_terms_list(as_strings=True, normalize=False))[0]
        )
        assert len(list(doc._.to_terms_list(ngrams=False))) < len(full_terms_list)
        assert len(list(doc._.to_terms_list(ngrams=1))) < len(full_terms_list)
        assert len(list(doc._.to_terms_list(ngrams=(1, 2)))) < len(full_terms_list)
        assert len(list(doc._.to_terms_list(ngrams=False))) < len(full_terms_list)

    def test_to_terms_list_kwargs(self, doc):
        full_terms_list = list(doc._.to_terms_list())
        for as_strings in (True, False):
            kwargs_sets = (
                {"ngrams": (1, 2), "filter_nums": True},
                {"ngrams": (1, 2), "entities": False},
                {"ngrams": (1, 2), "entities": None},
                {"normalize": "lower"},
                {"normalize": None},
                {"normalize": lambda term: term.text.upper()},
            )
            for kwargs in kwargs_sets:
                terms_list = list(doc._.to_terms_list(as_strings=as_strings, **kwargs))

    @pytest.mark.parametrize(
        "ngrams,entities,normalize,error",
        [
            (False, False, "lemma", pytest.raises(ValueError)),
            (False, (1, 2, 3), "lemma", pytest.raises(TypeError)),
            ((1, 2, 3), True, True, pytest.raises(ValueError)),
        ]
    )
    def test_to_terms_list_error(self, doc, ngrams, entities, normalize, error):
        with error:
            _ = list(
                doc._.to_terms_list(
                    ngrams=ngrams, entities=entities, normalize=normalize
                )
            )

    def test_to_bag_of_terms(self, doc):
        bot = doc._.to_bag_of_terms(weighting="count")
        assert isinstance(bot, dict)
        assert isinstance(list(bot.keys())[0], int)
        assert isinstance(list(bot.values())[0], int)
        bot = doc._.to_bag_of_terms(weighting="binary")
        assert isinstance(bot, dict)
        assert isinstance(list(bot.keys())[0], int)
        assert isinstance(list(bot.values())[0], int)
        for value in list(bot.values())[0:10]:
            assert value < 2
        bot = doc._.to_bag_of_terms(weighting="freq")
        assert isinstance(bot, dict)
        assert isinstance(list(bot.keys())[0], int)
        assert isinstance(list(bot.values())[0], float)
        bot = doc._.to_bag_of_terms(as_strings=True)
        assert isinstance(bot, dict)
        assert isinstance(list(bot.keys())[0], str)

    def test_to_bag_of_terms_error(self, doc):
        with pytest.raises(ValueError):
            _ = doc._.to_bag_of_terms(weighting="foo")

    def test_to_bag_of_words(self, doc):
        bow = doc._.to_bag_of_words(weighting="count")
        assert isinstance(bow, dict)
        assert isinstance(list(bow.keys())[0], int)
        assert isinstance(list(bow.values())[0], int)
        bow = doc._.to_bag_of_words(weighting="binary")
        assert isinstance(bow, dict)
        assert isinstance(list(bow.keys())[0], int)
        assert isinstance(list(bow.values())[0], int)
        for value in list(bow.values())[0:10]:
            assert value < 2
        bow = doc._.to_bag_of_words(weighting="freq")
        assert isinstance(bow, dict)
        assert isinstance(list(bow.keys())[0], int)
        assert isinstance(list(bow.values())[0], float)
        bow = doc._.to_bag_of_words(as_strings=True)
        assert isinstance(bow, dict)
        assert isinstance(list(bow.keys())[0], str)

    def test_to_bag_of_words_values(self):
        text = "Burton Jacob DeWilde, Burton Jacob, Burton."
        doc = make_spacy_doc(text, lang="en_core_web_sm")
        bow = doc._.to_bag_of_words(weighting="count", normalize="lower", as_strings=True)
        assert len(bow) == 3
        assert bow["burton"] == 3
        assert bow["jacob"] == 2
        assert bow["dewilde"] == 1
        bow = doc._.to_bag_of_words(weighting="freq", normalize="lower", as_strings=True)
        assert len(bow) == 3
        assert bow["burton"] == 3 / len(doc)
        assert bow["jacob"] == 2 / len(doc)
        assert bow["dewilde"] == 1 / len(doc)
        bow = doc._.to_bag_of_words(weighting="binary", normalize="lower", as_strings=True)
        assert len(bow) == 3
        assert bow["burton"] == 1
        assert bow["jacob"] == 1
        assert bow["dewilde"] == 1

    def test_to_bag_of_words_error(self, doc):
        with pytest.raises(ValueError):
            _ = doc._.to_bag_of_words(weighting="foo")

    def test_to_semantic_network_words(self, doc):
        graph = doc._.to_semantic_network(nodes="words", edge_weighting="cooc_freq")
        assert all(isinstance(node, str) for node in graph.nodes)
        assert all(isinstance(d["weight"], int) for n1, n2, d in graph.edges(data=True))
        graph = doc._.to_semantic_network(nodes="words", edge_weighting="binary")
        assert all(isinstance(node, str) for node in graph.nodes)
        assert all(d == {} for n1, n2, d in graph.edges(data=True))

    def test_to_semantic_network_sents(self, doc):
        graph = doc._.to_semantic_network(nodes="sents", edge_weighting="cosine")
        assert all(isinstance(node, int) for node in graph.nodes)
        assert all(isinstance(d["weight"], float) for n1, n2, d in graph.edges(data=True))
        graph = doc._.to_semantic_network(nodes="sents", edge_weighting="jaccard")
        assert all(isinstance(node, int) for node in graph.nodes)
        assert all(isinstance(d["weight"], int) for n1, n2, d in graph.edges(data=True))
