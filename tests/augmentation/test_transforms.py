# from spacy.tokens import Doc

from textacy import make_spacy_doc
from textacy.augmentation import transforms
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


@pytest.fixture(scope="module")
def doc_aug_toks(spacy_doc):
    return transforms.to_aug_toks(spacy_doc)


class TestSubstituteSynonyms:

    def test_noop(self, doc_aug_toks):
        for num in [0, 0.0]:
            for aug_toks in doc_aug_toks:
                new_aug_toks = transforms.substitute_synonyms(aug_toks, num)
                for aug_tok, new_aug_tok in zip(aug_toks, new_aug_toks):
                    assert aug_tok.text == new_aug_tok.text

    def test_num_int(self, doc_aug_toks):
        for num in [1, 3]:
            for aug_toks in doc_aug_toks:
                new_aug_toks = transforms.substitute_synonyms(aug_toks, num)
                assert isinstance(new_aug_toks, list)
                # this should be about the same, depending on the substitute synonyms
                # since some one-word tokens have two-word synonyms
                assert len(new_aug_toks) == pytest.approx(len(aug_toks), rel=0.05)
                assert all(
                    isinstance(aug_tok, transforms.AugTok)
                    for aug_tok in new_aug_toks
                )
                assert any(
                    aug_tok.text != new_aug_tok.text
                    for aug_tok, new_aug_tok in zip(aug_toks, new_aug_toks)
                )

    def test_num_float(self, doc_aug_toks):
        for num in [0.1, 0.3]:
            for aug_toks in doc_aug_toks:
                _ = transforms.substitute_synonyms(aug_toks, num)

    def test_errors(self, doc_aug_toks):
        for num in [-1, 2.0]:
            with pytest.raises(ValueError):
                _ = transforms.substitute_synonyms(doc_aug_toks[0], num)
        for obj in [["foo", "bar"], "foo bar"]:
            with pytest.raises(TypeError):
                _ = transforms.substitute_synonyms(obj, 1)


class TestInsertSynonyms:

    def test_noop(self, doc_aug_toks):
        for num in [0, 0.0]:
            for aug_toks in doc_aug_toks:
                new_aug_toks = transforms.insert_synonyms(aug_toks, num)
                for aug_tok, new_aug_tok in zip(aug_toks, new_aug_toks):
                    assert aug_tok.text == new_aug_tok.text

    def test_num_int(self, doc_aug_toks):
        for num in [1, 3]:
            for aug_toks in doc_aug_toks:
                new_aug_toks = transforms.insert_synonyms(aug_toks, num)
                assert isinstance(new_aug_toks, list)
                assert len(new_aug_toks) > len(aug_toks)
                assert all(
                    isinstance(aug_tok, transforms.AugTok)
                    for aug_tok in new_aug_toks
                )
                assert any(
                    aug_tok.text != new_aug_tok.text
                    for aug_tok, new_aug_tok in zip(aug_toks, new_aug_toks)
                )

    def test_num_float(self, doc_aug_toks):
        for num in [0.1, 0.3]:
            for aug_toks in doc_aug_toks:
                _ = transforms.insert_synonyms(aug_toks, num)

    def test_errors(self, doc_aug_toks):
        for num in [-1, 2.0]:
            with pytest.raises(ValueError):
                _ = transforms.insert_synonyms(doc_aug_toks[0], num)
        for obj in [["foo", "bar"], "foo bar"]:
            with pytest.raises(TypeError):
                _ = transforms.insert_synonyms(obj, 1)


class TestSwapTokens:

    def test_noop(self, doc_aug_toks):
        for num in [0, 0.0]:
            for aug_toks in doc_aug_toks:
                new_aug_toks = transforms.swap_tokens(aug_toks, num)
                for aug_tok, new_aug_tok in zip(aug_toks, new_aug_toks):
                    assert aug_tok.text == new_aug_tok.text

    def test_num_int(self, doc_aug_toks):
        for num in [1, 3]:
            for aug_toks in doc_aug_toks:
                new_aug_toks = transforms.swap_tokens(aug_toks, num)
                assert isinstance(new_aug_toks, list)
                assert len(new_aug_toks) == len(aug_toks)
                assert all(
                    isinstance(aug_tok, transforms.AugTok)
                    for aug_tok in new_aug_toks
                )
                assert any(
                    aug_tok.text != new_aug_tok.text
                    for aug_tok, new_aug_tok in zip(aug_toks, new_aug_toks)
                )

    def test_num_float(self, doc_aug_toks):
        for num in [0.1, 0.3]:
            for aug_toks in doc_aug_toks:
                _ = transforms.swap_tokens(aug_toks, num)

    def test_errors(self, doc_aug_toks):
        for num in [-1, 2.0]:
            with pytest.raises(ValueError):
                _ = transforms.swap_tokens(doc_aug_toks[0], num)
        for obj in [["foo", "bar"], "foo bar"]:
            with pytest.raises(TypeError):
                _ = transforms.swap_tokens(obj, 1)


class TestDeleteTokens:

    def test_noop(self, doc_aug_toks):
        for num in [0, 0.0]:
            for aug_toks in doc_aug_toks:
                new_aug_toks = transforms.delete_tokens(aug_toks, num)
                for aug_tok, new_aug_tok in zip(aug_toks, new_aug_toks):
                    assert aug_tok.text == new_aug_tok.text

    def test_num_int(self, doc_aug_toks):
        for num in [1, 3]:
            for aug_toks in doc_aug_toks:
                new_aug_toks = transforms.delete_tokens(aug_toks, num)
                assert isinstance(new_aug_toks, list)
                assert len(new_aug_toks) < len(aug_toks)
                assert all(
                    isinstance(aug_tok, transforms.AugTok)
                    for aug_tok in new_aug_toks
                )
                assert any(
                    aug_tok.text != new_aug_tok.text
                    for aug_tok, new_aug_tok in zip(aug_toks, new_aug_toks)
                )

    def test_num_float(self, doc_aug_toks):
        for num in [0.1, 0.3]:
            for aug_toks in doc_aug_toks:
                _ = transforms.delete_tokens(aug_toks, num)

    def test_errors(self, doc_aug_toks):
        for num in [-1, 2.0]:
            with pytest.raises(ValueError):
                _ = transforms.delete_tokens(doc_aug_toks[0], num)
        for obj in [["foo", "bar"], "foo bar"]:
            with pytest.raises(TypeError):
                _ = transforms.delete_tokens(obj, 1)

# class TestApply:

#     def test_noop(self, spacy_doc):
#         augmented_doc = transformations.apply(
#             spacy_doc,
#             n_replacements=0,
#             n_insertions=0,
#             n_swaps=0,
#             delete_prob=0.0,
#             shuffle_sents=False,
#         )
#         assert isinstance(augmented_doc, Doc)
#         assert augmented_doc.text == spacy_doc.text

#     def test_replace_with_synonyms(self, spacy_doc):
#         augmented_doc = transformations.apply(
#             spacy_doc,
#             n_replacements=2,
#             n_insertions=0,
#             n_swaps=0,
#             delete_prob=0.0,
#             shuffle_sents=False,
#         )
#         assert isinstance(augmented_doc, Doc)
#         assert augmented_doc.text != spacy_doc.text

#     def test_insert_synonyms(self, spacy_doc):
#         augmented_doc = transformations.apply(
#             spacy_doc,
#             n_replacements=0,
#             n_insertions=3,
#             n_swaps=0,
#             delete_prob=0.0,
#             shuffle_sents=False,
#         )
#         assert isinstance(augmented_doc, Doc)
#         assert augmented_doc.text != spacy_doc.text
#         # this is actually not guaranteed, depending on the synonyms
#         assert len(augmented_doc) >= len(spacy_doc)

#     def test_swap_items(self, spacy_doc):
#         augmented_doc = transformations.apply(
#             spacy_doc,
#             n_replacements=0,
#             n_insertions=0,
#             n_swaps=1,
#             delete_prob=0.0,
#             shuffle_sents=False,
#         )
#         assert isinstance(augmented_doc, Doc)
#         assert augmented_doc.text != spacy_doc.text
#         # this is actually not guaranteed, depending on the parser...
#         assert len(augmented_doc) == pytest.approx(len(spacy_doc), rel=0.01)

#     def test_delete_items(self, spacy_doc):
#         augmented_doc = transformations.apply(
#             spacy_doc,
#             n_replacements=0,
#             n_insertions=0,
#             n_swaps=0,
#             delete_prob=0.2,
#             shuffle_sents=False,
#         )
#         assert isinstance(augmented_doc, Doc)
#         assert augmented_doc.text != spacy_doc.text
#         # this is actually not guaranteed, depending on how random numbers play out
#         assert len(augmented_doc) < len(spacy_doc)

#     def test_shuffle_sents(self, spacy_doc):
#         augmented_doc = transformations.apply(
#             spacy_doc,
#             n_replacements=0,
#             n_insertions=0,
#             n_swaps=0,
#             delete_prob=0.0,
#             shuffle_sents=True,
#         )
#         assert isinstance(augmented_doc, Doc)
#         assert augmented_doc.text != spacy_doc.text
#         assert len(augmented_doc) == len(spacy_doc)
