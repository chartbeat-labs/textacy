import pytest

from textacy.augmentation import utils
from textacy.types import AugTok


def test_aug_tok():
    aug_tok = AugTok(text="text", ws=" ", pos="pos", is_word=True, syns=["doc"])
    assert isinstance(aug_tok, tuple)
    with pytest.raises(AttributeError):
        aug_tok.foo = "bar"


def test_to_aug_toks(doc_en):
    aug_toks = utils.to_aug_toks(doc_en)
    assert isinstance(aug_toks, list)
    assert all(isinstance(aug_tok, AugTok) for aug_tok in aug_toks)
    assert len(aug_toks) == len(doc_en)
    for obj in ["foo bar bat baz", ["foo", "bar", "bat", "baz"]]:
        with pytest.raises(TypeError):
            _ = utils.to_aug_toks(obj)


@pytest.mark.skipif(
    utils.udhr.index is None,
    reason="UDHR dataset must be downloaded before running this test",
)
def test_get_char_weights():
    for lang in ("en", "es", "xx"):
        char_weights = utils.get_char_weights(lang)
        assert isinstance(char_weights, list)
        assert all(isinstance(item, tuple) for item in char_weights)
        assert all(isinstance(char, str) for char, _ in char_weights)
        assert all(isinstance(weight, (int, float)) for _, weight in char_weights)
