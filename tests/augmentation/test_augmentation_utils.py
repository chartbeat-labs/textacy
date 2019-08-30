import pytest

from textacy import make_spacy_doc
from textacy.augmentation import utils


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


def test_aug_tok():
    aug_tok = utils.AugTok(text="text", ws=" ", pos="pos", is_word=True, syns=["doc"])
    assert isinstance(aug_tok, tuple)
    with pytest.raises(AttributeError):
        aug_tok.foo = "bar"


def test_to_aug_toks(spacy_doc):
    aug_toks = utils.to_aug_toks(spacy_doc)
    assert isinstance(aug_toks, list)
    assert all(isinstance(aug_tok, utils.AugTok) for aug_tok in aug_toks)
    assert len(aug_toks) == len(spacy_doc)
    for obj in ["foo bar bat baz", ["foo", "bar", "bat", "baz"]]:
        with pytest.raises(TypeError):
            _ = utils.to_aug_toks(obj)


@pytest.mark.skipif(
    utils.udhr.index is None,
    reason="UDHR dataset must be downloaded before running this test")
def test_get_char_weights():
    for lang in ("en", "es", "xx"):
        char_weights = utils.get_char_weights(lang)
        assert isinstance(char_weights, list)
        assert all(isinstance(item, tuple) for item in char_weights)
        assert all(isinstance(char, str) for char, _ in char_weights)
        assert all(isinstance(weight, (int, float)) for _, weight in char_weights)
