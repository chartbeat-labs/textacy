import functools

import pytest
from spacy.tokens import Doc

from textacy import make_spacy_doc
from textacy.augmentation import augmenter, transforms


@pytest.fixture(scope="module")
def spacy_doc():
    text = (
        "Democrats might know that they stand against Trump's policies, but coming up with their own plan is harder than you think. "
        "For a long time, the party's top echelon has been captive to free trade orthodoxy. "
        "Since Bill Clinton, the theory of the case among the Democratic Party's elite has been the more globalization, the better — with mostly a deaf ear turned to the people and places most badly affected. "
        "Worse, their response to globalization's excesses has been: "
        "Here's a new trade deal, much better than the last one."
    )
    return make_spacy_doc(text, lang="en_core_web_sm")


@pytest.fixture(scope="module")
def example_augmenter():
    return augmenter.Augmenter(
        [transforms.swap_words, transforms.delete_words, transforms.swap_chars, transforms.delete_chars],
        num=None,
    )


class TestAugmenter:

    def test_bad_args(self):
        for num in [-1, 10, -0.1, 2.0, [0.5, 0.5]]:
            with pytest.raises(ValueError):
                _ = augmenter.Augmenter([transforms.substitute_word_synonyms], num=num)
        for tfs in [[], ["not callable obj"]]:
            with pytest.raises((ValueError, TypeError)):
                _ = augmenter.Augmenter(tfs, num=1)

    def test_good_args(self):
        for num in [None, 1, 2, 0.25, [0.5, 0.5]]:
            _ = augmenter.Augmenter(
                [transforms.substitute_chars, transforms.insert_chars], num=num,
            )
        test_tfs = [
            [transforms.substitute_chars],
            [lambda x: x],
            [transforms.delete_chars, functools.partial(transforms.substitute_chars, num=1)],
        ]
        for tfs in test_tfs:
            _ = augmenter.Augmenter(tfs, num=1)

    def test_apply_transforms(self, spacy_doc, example_augmenter):
        new_doc1 = example_augmenter.apply_transforms(spacy_doc, lang="en_core_web_sm")
        new_doc2 = example_augmenter.apply_transforms(spacy_doc, lang="en_core_web_sm")
        assert isinstance(new_doc1, Doc)
        assert new_doc1.text != spacy_doc.text
        assert new_doc1.text != new_doc2.text
