import collections

from textacy import make_spacy_doc
from textacy.augmentation import transforms
from textacy.augmentation import utils as aug_utils

import pytest


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
def aug_toks(spacy_doc):
    return aug_utils.to_aug_toks(spacy_doc)


@pytest.mark.skipif(
    aug_utils.concept_net.filepath is None,
    reason="ConceptNet resource must be downloaded before running tests",
)
class TestSubstituteWordSynonyms:

    def test_noop(self, aug_toks):
        for num in [0, 0.0]:
            new_aug_toks = transforms.substitute_word_synonyms(aug_toks, num=num)
            for aug_tok, new_aug_tok in zip(aug_toks, new_aug_toks):
                assert aug_tok.text == new_aug_tok.text

    def test_num_int(self, aug_toks):
        for num in [1, 3]:
            new_aug_toks = transforms.substitute_word_synonyms(aug_toks, num=num)
            assert isinstance(new_aug_toks, list)
            assert len(new_aug_toks) == len(aug_toks)
            assert all(
                isinstance(aug_tok, aug_utils.AugTok)
                for aug_tok in new_aug_toks
            )
            assert any(
                aug_tok.text != new_aug_tok.text
                for aug_tok, new_aug_tok in zip(aug_toks, new_aug_toks)
            )

    def test_num_float(self, aug_toks):
        for num in [0.1, 0.3]:
            _ = transforms.substitute_word_synonyms(aug_toks, num=num)

    def test_pos(self, aug_toks):
        for pos in ["NOUN", ("NOUN", "VERB", "ADJ", "ADV")]:
            new_aug_toks = transforms.substitute_word_synonyms(
                aug_toks, num=1, pos=pos)
            assert isinstance(new_aug_toks, list)
            assert len(new_aug_toks) == len(aug_toks)
            assert all(
                isinstance(aug_tok, aug_utils.AugTok)
                for aug_tok in new_aug_toks
            )
            assert any(
                aug_tok.text != new_aug_tok.text
                for aug_tok, new_aug_tok in zip(aug_toks, new_aug_toks)
            )

    def test_errors(self, aug_toks):
        for num in [-1, 2.0]:
            with pytest.raises(ValueError):
                _ = transforms.substitute_word_synonyms(aug_toks, num=num)
        for obj in [["foo", "bar"], "foo bar"]:
            with pytest.raises(TypeError):
                _ = transforms.substitute_word_synonyms(obj, num=1)


@pytest.mark.skipif(
    aug_utils.concept_net.filepath is None,
    reason="ConceptNet resource must be downloaded before running tests",
)
class TestInsertWordSynonyms:

    def test_noop(self, aug_toks):
        for num in [0, 0.0]:
            new_aug_toks = transforms.insert_word_synonyms(aug_toks, num=num)
            for aug_tok, new_aug_tok in zip(aug_toks, new_aug_toks):
                assert aug_tok.text == new_aug_tok.text

    def test_num_int(self, aug_toks):
        for num in [1, 3]:
            new_aug_toks = transforms.insert_word_synonyms(aug_toks, num=num)
            assert isinstance(new_aug_toks, list)
            assert len(new_aug_toks) > len(aug_toks)
            assert all(
                isinstance(aug_tok, aug_utils.AugTok)
                for aug_tok in new_aug_toks
            )

    def test_num_float(self, aug_toks):
        for num in [0.1, 0.3]:
            _ = transforms.insert_word_synonyms(aug_toks, num=num)

    def test_pos(self, aug_toks):
        for pos in ["NOUN", ("NOUN", "VERB", "ADJ", "ADV")]:
            new_aug_toks = transforms.insert_word_synonyms(
                aug_toks, num=1, pos=pos)
            assert isinstance(new_aug_toks, list)
            assert len(new_aug_toks) > len(aug_toks)
            assert all(
                isinstance(aug_tok, aug_utils.AugTok)
                for aug_tok in new_aug_toks
            )

    def test_errors(self, aug_toks):
        for num in [-1, 2.0]:
            with pytest.raises(ValueError):
                _ = transforms.insert_word_synonyms(aug_toks, num=num)
        for obj in [["foo", "bar"], "foo bar"]:
            with pytest.raises(TypeError):
                _ = transforms.insert_word_synonyms(obj, num=1)


class TestSwapWords:

    def test_noop(self, aug_toks):
        for num in [0, 0.0]:
            new_aug_toks = transforms.swap_words(aug_toks, num=num)
            for aug_tok, new_aug_tok in zip(aug_toks, new_aug_toks):
                assert aug_tok.text == new_aug_tok.text

    def test_num_int(self, aug_toks):
        for num in [1, 3]:
            new_aug_toks = transforms.swap_words(aug_toks, num=num)
            assert isinstance(new_aug_toks, list)
            assert len(new_aug_toks) == len(aug_toks)
            assert all(
                isinstance(aug_tok, aug_utils.AugTok)
                for aug_tok in new_aug_toks
            )
            assert any(
                aug_tok.text != new_aug_tok.text
                for aug_tok, new_aug_tok in zip(aug_toks, new_aug_toks)
            )

    def test_num_float(self, aug_toks):
        for num in [0.1, 0.3]:
            _ = transforms.swap_words(aug_toks, num=num)

    def test_pos(self, aug_toks):
        for pos in ["NOUN", ("NOUN", "VERB", "ADJ", "ADV")]:
            new_aug_toks = transforms.swap_words(aug_toks, num=1, pos=pos)
            assert isinstance(new_aug_toks, list)
            assert len(new_aug_toks) == len(aug_toks)
            assert all(
                isinstance(aug_tok, aug_utils.AugTok)
                for aug_tok in new_aug_toks
            )
            assert any(
                aug_tok.text != new_aug_tok.text
                for aug_tok, new_aug_tok in zip(aug_toks, new_aug_toks)
            )

    def test_errors(self, aug_toks):
        for num in [-1, 2.0]:
            with pytest.raises(ValueError):
                _ = transforms.swap_words(aug_toks, num=num)
        for obj in [["foo", "bar"], "foo bar"]:
            with pytest.raises(TypeError):
                _ = transforms.swap_words(obj, num=1)


class TestDeleteWords:

    def test_noop(self, aug_toks):
        for num in [0, 0.0]:
            new_aug_toks = transforms.delete_words(aug_toks, num=num)
            for aug_tok, new_aug_tok in zip(aug_toks, new_aug_toks):
                assert aug_tok.text == new_aug_tok.text

    def test_num_int(self, aug_toks):
        for num in [1, 3]:
            new_aug_toks = transforms.delete_words(aug_toks, num=num)
            assert isinstance(new_aug_toks, list)
            assert len(new_aug_toks) < len(aug_toks)
            assert all(
                isinstance(aug_tok, aug_utils.AugTok)
                for aug_tok in new_aug_toks
            )

    def test_num_float(self, aug_toks):
        for num in [0.1, 0.3]:
            _ = transforms.delete_words(aug_toks, num=num)

    def test_pos(self, aug_toks):
        for pos in ["NOUN", ("NOUN", "VERB", "ADJ", "ADV")]:
            new_aug_toks = transforms.delete_words(aug_toks, num=1, pos=pos)
            assert isinstance(new_aug_toks, list)
            assert len(new_aug_toks) < len(aug_toks)
            assert all(
                isinstance(aug_tok, aug_utils.AugTok)
                for aug_tok in new_aug_toks
            )

    def test_errors(self, aug_toks):
        for num in [-1, 2.0]:
            with pytest.raises(ValueError):
                _ = transforms.delete_words(aug_toks, num=num)
        for obj in [["foo", "bar"], "foo bar"]:
            with pytest.raises(TypeError):
                _ = transforms.delete_words(obj, num=1)


@pytest.mark.skipif(
    aug_utils.udhr.index is None,
    reason="UDHR dataset must be downloaded before running tests",
)
class TestSubstituteChars:

    def test_noop(self, aug_toks):
        for num in [0, 0.0]:
            new_aug_toks = transforms.substitute_chars(aug_toks, num=num, lang="en")
            for aug_tok, new_aug_tok in zip(aug_toks, new_aug_toks):
                assert aug_tok.text == new_aug_tok.text

    def test_num_int(self, aug_toks):
        # using higher nums here to prevent the very unlikely case
        # that all characters are substituted by the same character
        for num in [3, 5]:
            new_aug_toks = transforms.substitute_chars(aug_toks, num=num, lang="en")
            assert isinstance(new_aug_toks, list)
            assert len(new_aug_toks) == len(aug_toks)
            assert all(
                isinstance(aug_tok, aug_utils.AugTok)
                for aug_tok in new_aug_toks
            )
            assert any(
                aug_tok.text != new_aug_tok.text
                for aug_tok, new_aug_tok in zip(aug_toks, new_aug_toks)
            )
            assert all(
                len(aug_tok.text) == len(new_aug_tok.text)
                for aug_tok, new_aug_tok in zip(aug_toks, new_aug_toks)
            )

    def test_num_float(self, aug_toks):
        for num in [0.1, 0.3]:
            _ = transforms.substitute_chars(aug_toks, num=num, lang="en")

    def test_errors(self, aug_toks):
        for num in [-1, 2.0]:
            with pytest.raises(ValueError):
                _ = transforms.substitute_chars(aug_toks, num=num, lang="en")
        for obj in [["foo", "bar"], "foo bar"]:
            with pytest.raises(TypeError):
                _ = transforms.substitute_chars(obj, num=1, lang="en")


@pytest.mark.skipif(
    aug_utils.udhr.index is None,
    reason="UDHR dataset must be downloaded before running tests",
)
class TestInsertChars:

    def test_noop(self, aug_toks):
        for num in [0, 0.0]:
            new_aug_toks = transforms.insert_chars(aug_toks, num=num, lang="en")
            for aug_tok, new_aug_tok in zip(aug_toks, new_aug_toks):
                assert aug_tok.text == new_aug_tok.text

    def test_num_int(self, aug_toks):
        for num in [1, 3]:
            new_aug_toks = transforms.insert_chars(aug_toks, num=num, lang="en")
            assert isinstance(new_aug_toks, list)
            assert len(new_aug_toks) == len(aug_toks)
            assert all(
                isinstance(aug_tok, aug_utils.AugTok)
                for aug_tok in new_aug_toks
            )
            assert all(
                (
                    aug_tok.text == new_aug_tok.text or
                    len(aug_tok.text) < len(new_aug_tok.text)
                )
                for aug_tok, new_aug_tok in zip(aug_toks, new_aug_toks)
            )

    def test_num_float(self, aug_toks):
        for num in [0.1, 0.3]:
            _ = transforms.insert_chars(aug_toks, num=num, lang="en")

    def test_errors(self, aug_toks):
        for num in [-1, 2.0]:
            with pytest.raises(ValueError):
                _ = transforms.insert_chars(aug_toks, num=num, lang="en")
        for obj in [["foo", "bar"], "foo bar"]:
            with pytest.raises(TypeError):
                _ = transforms.insert_chars(obj, num=1, lang="en")


class TestSwapChars:

    def test_noop(self, aug_toks):
        for num in [0, 0.0]:
            new_aug_toks = transforms.swap_chars(aug_toks, num=num)
            for aug_tok, new_aug_tok in zip(aug_toks, new_aug_toks):
                assert aug_tok.text == new_aug_tok.text

    def test_num_int(self, aug_toks):
        # using higher nums here to prevent the very unlikely case
        # that all characters are swapped with the same character
        for num in [3, 5]:
            new_aug_toks = transforms.swap_chars(aug_toks, num=num)
            assert isinstance(new_aug_toks, list)
            assert len(new_aug_toks) == len(aug_toks)
            assert all(
                isinstance(aug_tok, aug_utils.AugTok)
                for aug_tok in new_aug_toks
            )
            assert any(
                aug_tok.text != new_aug_tok.text
                for aug_tok, new_aug_tok in zip(aug_toks, new_aug_toks)
            )
            assert all(
                len(aug_tok.text) == len(new_aug_tok.text)
                for aug_tok, new_aug_tok in zip(aug_toks, new_aug_toks)
            )

    def test_num_float(self, aug_toks):
        for num in [0.1, 0.3]:
            _ = transforms.swap_chars(aug_toks, num=num)

    def test_errors(self, aug_toks):
        for num in [-1, 2.0]:
            with pytest.raises(ValueError):
                _ = transforms.swap_chars(aug_toks, num=num)
        for obj in [["foo", "bar"], "foo bar"]:
            with pytest.raises(TypeError):
                _ = transforms.swap_chars(obj, num=1)


class TestDeleteChars:

    def test_noop(self, aug_toks):
        for num in [0, 0.0]:
            new_aug_toks = transforms.delete_chars(aug_toks, num=num)
            for aug_tok, new_aug_tok in zip(aug_toks, new_aug_toks):
                assert aug_tok.text == new_aug_tok.text

    def test_num_int(self, aug_toks):
        for num in [1, 3]:
            new_aug_toks = transforms.delete_chars(aug_toks, num=num)
            assert isinstance(new_aug_toks, list)
            assert len(new_aug_toks) == len(aug_toks)
            assert all(
                isinstance(aug_tok, aug_utils.AugTok)
                for aug_tok in new_aug_toks
            )
            assert all(
                (
                    aug_tok.text == new_aug_tok.text or
                    len(aug_tok.text) > len(new_aug_tok.text)
                )
                for aug_tok, new_aug_tok in zip(aug_toks, new_aug_toks)
            )

    def test_num_float(self, aug_toks):
        for num in [0.1, 0.3]:
            _ = transforms.delete_chars(aug_toks, num=num)

    def test_errors(self, aug_toks):
        for num in [-1, 2.0]:
            with pytest.raises(ValueError):
                _ = transforms.delete_chars(aug_toks, num=num)
        for obj in [["foo", "bar"], "foo bar"]:
            with pytest.raises(TypeError):
                _ = transforms.delete_chars(obj, num=1)
