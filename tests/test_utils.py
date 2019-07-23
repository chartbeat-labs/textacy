import pytest

from textacy import utils

STRINGS = (b"bytes", "unicode", "úñîçødé")
NOT_STRINGS = (1, 2.0, ["foo", "bar"], {"foo": "bar"})


def test_to_collection():
    in_outs = [
        [(1, int, list), [1]],
        [([1, 2], int, tuple), (1, 2)],
        [((1, 1.0), (int, float), set), {1, 1.0}],
    ]
    assert utils.to_collection(None, int, list) is None
    for in_, out_ in in_outs:
        assert utils.to_collection(*in_) == out_


def test_to_unicode():
    for obj in STRINGS:
        assert isinstance(utils.to_unicode(obj), str)
    for obj in NOT_STRINGS:
        with pytest.raises(TypeError):
            utils.to_unicode(obj)


def test_to_bytes():
    for obj in STRINGS:
        assert isinstance(utils.to_bytes(obj), bytes)
    for obj in NOT_STRINGS:
        with pytest.raises(TypeError):
            utils.to_bytes(obj)
