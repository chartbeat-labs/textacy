import functools

import pytest
from spacy.tokens import Doc

from textacy.augmentation import augmenter, transforms


@pytest.fixture(scope="module")
def example_augmenter():
    return augmenter.Augmenter(
        [
            transforms.swap_words,
            transforms.delete_words,
            transforms.swap_chars,
            transforms.delete_chars,
        ],
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
                [transforms.substitute_chars, transforms.insert_chars],
                num=num,
            )
        test_tfs = [
            [transforms.substitute_chars],
            [lambda x: x],
            [
                transforms.delete_chars,
                functools.partial(transforms.substitute_chars, num=1),
            ],
        ]
        for tfs in test_tfs:
            _ = augmenter.Augmenter(tfs, num=1)

    def test_apply_transforms(self, doc_en, example_augmenter):
        new_doc1 = example_augmenter.apply_transforms(doc_en, lang="en_core_web_sm")
        new_doc2 = example_augmenter.apply_transforms(doc_en, lang="en_core_web_sm")
        assert isinstance(new_doc1, Doc)
        assert new_doc1.text != doc_en.text
        assert new_doc1.text != new_doc2.text
