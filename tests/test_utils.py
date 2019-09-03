import datetime
import pathlib

import pytest

from textacy import utils


PATHS = (pathlib.Path("."), pathlib.Path.home())
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


def test_to_path():
    for obj in PATHS:
        assert isinstance(utils.to_path(obj), pathlib.Path)
    for obj in STRINGS:
        if isinstance(obj, str):
            assert isinstance(utils.to_path(obj), pathlib.Path)
    for obj in NOT_STRINGS:
        with pytest.raises(TypeError):
            utils.to_path(obj)


class TestValidateAndClipRange:

    def test_good_inputs(self):
        inputs = [
            [("2001-01", "2002-01"), ("2000-01", "2003-01")],
            [["2001-01", "2004-01"], ("2000-01", "2003-01")],
            [("2001-01", "2002-01"), ["2000-01", "2003-01"], (str, bytes)],
            [[-5, 5], [-10, 10]],
            [(-5, 5), (0, 10)],
            [(-5, 5), (-10, 10), int],
            [(-5, 5), (-10, 10), (int, float)],
        ]
        for input_ in inputs:
            output = utils.validate_and_clip_range(*input_)
            assert isinstance(output, tuple)
            assert len(output) == 2
            assert output[0] == max(input_[0][0], input_[1][0])
            assert output[1] == min(input_[0][1], input_[1][1])

    def test_null_inputs(self):
        inputs = [
            [(0, None), (-5, 5)],
            [(None, 0), (-5, 5)],
        ]
        for input_ in inputs:
            output = utils.validate_and_clip_range(*input_)
            assert isinstance(output, tuple)
            assert len(output) == 2
            if input_[0][0] is None:
                assert output[0] == input_[1][0]
            elif input_[0][1] is None:
                assert output[1] == input_[1][1]

    def test_bad_typeerror(self):
        inputs = [
            ["2001-01", ("2000-01", "2003-01")],
            [("2001-01", "2002-01"), "2000-01"],
            [{"2001-01", "2002-01"}, ("2000-01", "2003-01")],
            [("2001-01", "2002-01"), ("2000-01", "2003-01"), datetime.date],
            [0, [-10, 10]],
            [(-5, 5), 0],
            [[-5, 5], [-10, 10], (str, bytes)],
        ]
        for input_ in inputs:
            with pytest.raises(TypeError):
                utils.validate_and_clip_range(*input_)

    def test_bad_valueerror(self):
        inputs = [
            [("2001-01", "2002-01", "2003-01"), ("2000-01", "2003-01")],
            [("2001-01", "2002-01"), ["2000-01", "2002-01", "2004-01"]],
            [[0, 5, 10], (-10, 10)],
            [(-5, 5), [-10, 0, 10]],
            [(-5, 5), [-10, 0, 10], (str, bytes)],
        ]
        for input_ in inputs:
            with pytest.raises(ValueError):
                utils.validate_and_clip_range(*input_)


class TestValidateSetMembers:

    def test_good_inputs(self):
        inputs = [
            [{"a", "b"}, (str, bytes), {"a", "b", "c"}],
            ["a", (str, bytes), {"a", "b", "c"}],
            [("a", "b"), (str, bytes), {"a", "b", "c"}],
            [["a", "b"], (str, bytes)],
            [{1, 2}, int, {1, 2, 3}],
            [{1, 2}, (int, float), {1, 2, 3}],
            [1, int, {1: "a", 2: "b", 3: "c"}],
            [{3.14, 42.0}, float],
            [3.14, (int, float)],
        ]
        for input_ in inputs:
            output = utils.validate_set_members(*input_)
            assert isinstance(output, set)
            assert all(isinstance(val, input_[1]) for val in output)

    def test_bad_typeerror(self):
        inputs = [
            [{"a", "b"}, int],
            ["a", int],
            [("a", "b"), (int, float)],
        ]
        for input_ in inputs:
            with pytest.raises(TypeError):
                utils.validate_set_members(*input_)

    def test_bad_valueerror(self):
        inputs = [
            [{"a", "b"}, (str, bytes), {"x", "y", "z"}],
            [{"a", "x"}, (str, bytes), {"x", "y", "z"}],
            ["a", (str, bytes), {"x", "y", "z"}],
            ["a", (str, bytes), {"x": 24, "y": 25, "z": 26}],
        ]
        for input_ in inputs:
            with pytest.raises(ValueError):
                utils.validate_set_members(*input_)
