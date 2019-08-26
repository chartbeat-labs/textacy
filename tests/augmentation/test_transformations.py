from spacy.tokens import Doc

from textacy import make_spacy_doc
from textacy.augmentation import transformations
from textacy.resources import ConceptNet

import pytest


RESOURCE = ConceptNet()

pytestmark = pytest.mark.skipif(
    RESOURCE.filepath is None,
    reason="ConceptNet resource must be downloaded before running tests",
)

@pytest.fixture(scope="module")
def spacy_doc():
    text = (
        "Democrats might know that they stand against Trump's policies, but coming up with their own plan is harder than you think. "
        "For a long time, the party's top echelon has been captive to free trade orthodoxy. "
        "Since Bill Clinton, the theory of the case among the Democratic Party's elite has been the more globalization, the better — with mostly a deaf ear turned to the people and places most badly affected. "
        "Worse, their response to globalization's excesses has been: "
        "Here's a new trade deal, much better than the last one."
    )
    return make_spacy_doc(text, lang="en")


class TestApply:

    def test_noop(self, spacy_doc):
        augmented_doc = transformations.apply(
            spacy_doc,
            n_replacements=0,
            n_insertions=0,
            n_swaps=0,
            delete_prob=0.0,
            shuffle_sents=False,
        )
        assert isinstance(augmented_doc, Doc)
        assert augmented_doc.text == spacy_doc.text

    def test_replace_with_synonyms(self, spacy_doc):
        augmented_doc = transformations.apply(
            spacy_doc,
            n_replacements=2,
            n_insertions=0,
            n_swaps=0,
            delete_prob=0.0,
            shuffle_sents=False,
        )
        assert isinstance(augmented_doc, Doc)
        assert augmented_doc.text != spacy_doc.text

    def test_insert_synonyms(self, spacy_doc):
        augmented_doc = transformations.apply(
            spacy_doc,
            n_replacements=0,
            n_insertions=3,
            n_swaps=0,
            delete_prob=0.0,
            shuffle_sents=False,
        )
        assert isinstance(augmented_doc, Doc)
        assert augmented_doc.text != spacy_doc.text
        # this is actually not guaranteed, depending on the synonyms
        assert len(augmented_doc) >= len(spacy_doc)

    def test_swap_items(self, spacy_doc):
        augmented_doc = transformations.apply(
            spacy_doc,
            n_replacements=0,
            n_insertions=0,
            n_swaps=1,
            delete_prob=0.0,
            shuffle_sents=False,
        )
        assert isinstance(augmented_doc, Doc)
        assert augmented_doc.text != spacy_doc.text
        # this is actually not guaranteed, depending on the parser...
        assert len(augmented_doc) == pytest.approx(len(spacy_doc), rel=0.01)

    def test_delete_items(self, spacy_doc):
        augmented_doc = transformations.apply(
            spacy_doc,
            n_replacements=0,
            n_insertions=0,
            n_swaps=0,
            delete_prob=0.2,
            shuffle_sents=False,
        )
        assert isinstance(augmented_doc, Doc)
        assert augmented_doc.text != spacy_doc.text
        # this is actually not guaranteed, depending on how random numbers play out
        assert len(augmented_doc) < len(spacy_doc)

    def test_shuffle_sents(self, spacy_doc):
        augmented_doc = transformations.apply(
            spacy_doc,
            n_replacements=0,
            n_insertions=0,
            n_swaps=0,
            delete_prob=0.0,
            shuffle_sents=True,
        )
        assert isinstance(augmented_doc, Doc)
        assert augmented_doc.text != spacy_doc.text
        assert len(augmented_doc) == len(spacy_doc)
