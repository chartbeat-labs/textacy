from cytoolz import itertoolz
import pytest

import textacy
from textacy.representations import network


@pytest.fixture(scope="module")
def docs():
    lang = textacy.load_spacy_lang("en_core_web_sm")
    texts = [
        "Mary had a little lamb. Its fleece was white as snow.",
        "Everywhere that Mary went the lamb was sure to go.",
        # "It followed her to school one day, which was against the rule.",
        # "It made the children laugh and play to see a lamb at school.",
        # "And so the teacher turned it out, but still it lingered near.",
        # "It waited patiently about until Mary did appear.",
        # "Why does the lamb love Mary so? The eager children cry.",
        # "Mary loves the lamb, you know, the teacher did reply.",
    ]
    return [textacy.make_spacy_doc(text, lang=lang) for text in texts]


class TestCooccurrenceNetwork:

    def test_data_sequence_str(self, docs):
        data = [tok.text for tok in docs[0]]
        graph = network.build_cooccurrence_network(data)
        assert len(graph) == len(set(data))
        assert all("weight" in data for _, _, data in graph.edges(data=True))

    def test_data_sequence_sequence_str(self, docs):
        data = [[tok.lower_ for tok in doc] for doc in docs]
        graph = network.build_cooccurrence_network(data)
        assert len(graph) == len(set(itertoolz.concat(data)))
        assert all("weight" in data for _, _, data in graph.edges(data=True))

    def test_window_size(self, docs):
        data = [tok.text for tok in docs[0]]
        graph_ws2 = network.build_cooccurrence_network(data, window_size=2)
        graph_ws3 = network.build_cooccurrence_network(data, window_size=3)
        assert len(graph_ws2) == len(graph_ws3) == len(set(data))
        assert(
            sum(data["weight"] for _, _, data in graph_ws2.edges(data=True)) <
            sum(data["weight"] for _, _, data in graph_ws3.edges(data=True))
        )

    @pytest.mark.parametrize("edge_weighting", ["count", "binary"])
    def test_edge_weighting(self, edge_weighting, docs):
        data = [tok.text for tok in docs[0]]
        graph = network.build_cooccurrence_network(data, edge_weighting=edge_weighting)
        assert len(graph) == len(set(data))
        if edge_weighting == "count":
            assert(
                all(
                    isinstance(data["weight"], int) and data["weight"] >= 1
                    for _, _, data in graph.edges(data=True)
                )
            )
        elif edge_weighting == "binary":
            assert(all(data["weight"] == 1 for _, _, data in graph.edges(data=True)))


class TestSimilarityNetwork:

    def test_data_sequence_str(self, docs):
        data = [tok.text for tok in docs[0]]
        graph = network.build_similarity_network(data, "levenshtein")
        assert len(graph) == len(set(data))
        assert all("weight" in data for _, _, data in graph.edges(data=True))

    def test_data_sequence_sequence_str(self, docs):
        data = [tuple(tok.lower_ for tok in doc) for doc in docs]
        graph = network.build_similarity_network(data, "jaccard")
        assert len(graph) == len(set(data))
        assert all("weight" in data for _, _, data in graph.edges(data=True))

    @pytest.mark.parametrize("edge_weighting", ["hamming", "levenshtein", "jaro"])
    def test_edit_similarity_metrics(self, edge_weighting, docs):
        data = [tok.text for tok in docs[0]]
        graph = network.build_similarity_network(data, edge_weighting)
        assert len(graph) == len(set(data))
        assert(
            all(
                isinstance(data["weight"], float) and 0.0 <= data["weight"] <= 1.0
                for _, _, data in graph.edges(data=True)
            )
        )

    @pytest.mark.parametrize("edge_weighting", ["jaccard", "cosine", "bag"])
    def test_tok_similarity_metrics(self, edge_weighting, docs):
        data = [tuple(tok.lower_ for tok in doc) for doc in docs]
        graph = network.build_similarity_network(data, edge_weighting)
        assert len(graph) == len(set(data))
        assert(
            all(
                isinstance(data["weight"], float) and 0.0 <= data["weight"] <= 1.0
                for _, _, data in graph.edges(data=True)
            )
        )
