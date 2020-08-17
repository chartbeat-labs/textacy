import datetime
import pathlib

import pytest

from textacy import utils


@pytest.mark.parametrize(
    "val,val_type,col_type,expected",
    [
        (None, int, list, None),
        (1, int, list, [1]),
        ([1, 2], int, tuple, (1, 2)),
        ((1, 1.0), (int, float), set, {1, 1.0}),
    ],
)
def test_to_collection(val, val_type, col_type, expected):
    assert utils.to_collection(val, val_type, col_type) == expected


class TestToUnicode:

    @pytest.mark.parametrize("s", [b"bytes", "unicode", "úñîçødé"])
    def test_valid(self, s):
        assert isinstance(utils.to_unicode(s), str)

    @pytest.mark.parametrize("s", [1, 2.0, ["foo", "bar"], {"foo": "bar"}])
    def test_invalid(self, s):
        with pytest.raises(TypeError):
            _ = utils.to_unicode(s)


class TestToBytes:

    @pytest.mark.parametrize("s", [b"bytes", "unicode", "úñîçødé"])
    def test_valid(self, s):
        assert isinstance(utils.to_bytes(s), bytes)

    @pytest.mark.parametrize("s", [1, 2.0, ["foo", "bar"], {"foo": "bar"}])
    def test_invalid(self, s):
        with pytest.raises(TypeError):
            _ = utils.to_bytes(s)


class TestToPath:

    @pytest.mark.parametrize("path", [pathlib.Path("."), pathlib.Path.home()])
    def test_path_input(self, path):
        assert isinstance(utils.to_path(path), pathlib.Path)

    @pytest.mark.parametrize("path", ["unicode", "úñîçødé"])
    def test_str_input(self, path):
        assert isinstance(utils.to_path(path), pathlib.Path)

    @pytest.mark.parametrize("path", [1, 2.0, ["foo", "bar"], {"foo": "bar"}])
    def test_invalid_input(self, path):
        with pytest.raises(TypeError):
            _ = utils.to_path(path)


class TestValidateAndClipRange:

    @pytest.mark.parametrize(
        "range_vals,full_range,val_type",
        [
            [("2001-01", "2002-01"), ("2000-01", "2003-01"), None],
            [["2001-01", "2004-01"], ("2000-01", "2003-01"), None],
            [("2001-01", "2002-01"), ["2000-01", "2003-01"], (str, bytes)],
            [[-5, 5], [-10, 10], None],
            [(-5, 5), (0, 10), None],
            [(-5, 5), (-10, 10), int],
            [(-5, 5), (-10, 10), (int, float)],
            [(0, None), (-5, 5), None],
            [(None, 0), (-5, 5), None],
        ],
    )
    def test_valid_inputs(self, range_vals, full_range, val_type):
        output = utils.validate_and_clip_range(range_vals, full_range, val_type)
        assert isinstance(output, tuple)
        assert len(output) == 2
        if range_vals[0] is None:
            assert output[0] == full_range[0]
        else:
            assert output[0] == max(range_vals[0], full_range[0])
        if range_vals[1] is None:
            assert output[1] == full_range[1]
        else:
            assert output[1] == min(range_vals[1], full_range[1])

    @pytest.mark.parametrize(
        "range_vals,full_range,val_type,error",
        [
            ["2001-01", ("2000-01", "2003-01"), None, pytest.raises(TypeError)],
            [("2001-01", "2002-01"), "2000-01", None, pytest.raises(TypeError)],
            [
                {"2001-01", "2002-01"},
                ("2000-01", "2003-01"),
                None,
                pytest.raises(TypeError),
            ],
            [
                ("2001-01", "2002-01"),
                ("2000-01", "2003-01"),
                datetime.date,
                pytest.raises(TypeError),
            ],
            [0, [-10, 10], None, pytest.raises(TypeError)],
            [(-5, 5), 0, None, pytest.raises(TypeError)],
            [[-5, 5], [-10, 10], (str, bytes), pytest.raises(TypeError)],
            [
                ("2001-01", "2002-01", "2003-01"),
                ("2000-01", "2003-01"),
                None,
                pytest.raises(ValueError),
            ],
            [
                ("2001-01", "2002-01"),
                ["2000-01", "2002-01", "2004-01"],
                None,
                pytest.raises(ValueError),
            ],
            [[0, 5, 10], (-10, 10), None, pytest.raises(ValueError)],
            [(-5, 5), [-10, 0, 10], None, pytest.raises(ValueError)],
            [(-5, 5), [-10, 0, 10], (str, bytes), pytest.raises(ValueError)],
        ],
    )
    def test_invalid_inputs(self, range_vals, full_range, val_type, error):
        with error:
            _ = utils.validate_and_clip_range(range_vals, full_range, val_type)


class TestValidateSetMembers:

    @pytest.mark.parametrize(
        "vals,val_type,valid_vals",
        [
            [{"a", "b"}, (str, bytes), {"a", "b", "c"}],
            ["a", (str, bytes), {"a", "b", "c"}],
            [("a", "b"), (str, bytes), {"a", "b", "c"}],
            [["a", "b"], (str, bytes), None],
            [{1, 2}, int, {1, 2, 3}],
            [{1, 2}, (int, float), {1, 2, 3}],
            [1, int, {1: "a", 2: "b", 3: "c"}],
            [{3.14, 42.0}, float, None],
            [3.14, (int, float), None],
        ]
    )
    def test_valid_inputs(self, vals, val_type, valid_vals):
        output = utils.validate_set_members(vals, val_type, valid_vals)
        assert isinstance(output, set)
        assert all(isinstance(val, val_type) for val in output)


    @pytest.mark.parametrize(
        "vals,val_type,valid_vals,error",
        [
            [{"a", "b"}, int, None, pytest.raises(TypeError)],
            ["a", int, None, pytest.raises(TypeError)],
            [("a", "b"), (int, float), None, pytest.raises(TypeError)],
            [{"a", "b"}, (str, bytes), {"x", "y", "z"}, pytest.raises(ValueError)],
            [{"a", "x"}, (str, bytes), {"x", "y", "z"}, pytest.raises(ValueError)],
            ["a", (str, bytes), {"x", "y", "z"}, pytest.raises(ValueError)],
            ["a", (str, bytes), {"x": 24, "y": 25, "z": 26}, pytest.raises(ValueError)],
        ]
    )
    def test_invalid_inputs(self, vals, val_type, valid_vals, error):
        with error:
            _ = utils.validate_set_members(vals, val_type, valid_vals)



# TODO: uncomment this when we're only supporting PY3.8+
# def _func_pos_only_args(parg1, parg2, /):
#     return (parg1, parg2)


# TODO: uncomment this when we're only supporting PY3.8+
# def _func_mix_args(parg, /, arg, *, kwarg):
#     return (parg, arg, kwarg)


def _func_mix_args(arg, *, kwarg):
    return (arg, kwarg)


def _func_kw_only_args(*, kwarg1, kwarg2):
    return (kwarg1, kwarg2)


@pytest.mark.parametrize(
    "func,kwargs,expected",
    [
        # (_func_pos_only_args, {"kwarg": "kwargval"}, {}),
        (_func_mix_args, {"arg": "argval"}, {"arg": "argval"}),
        (
            _func_mix_args,
            {"arg": "argval", "kwarg": "kwarval"},
            {"arg": "argval", "kwarg": "kwarval"},
        ),
        (
            _func_mix_args,
            {"arg": "argval", "kwarg": "kwargval", "foo": "bar"},
            {"arg": "argval", "kwarg": "kwargval"},
        ),
        (
            _func_kw_only_args,
            {"kwarg1": "kwarg1val", "kwarg2": "kwarg2val"},
            {"kwarg1": "kwarg1val", "kwarg2": "kwarg2val"},
        ),
        (
            _func_kw_only_args,
            {"kwarg1": "kwarg1val", "kwarg3": "kwarg3val"},
            {"kwarg1": "kwarg1val"},
        ),
        (_func_kw_only_args, {}, {}),
    ],
)
def test_get_kwargs_for_func(func, kwargs, expected):
    assert utils.get_kwargs_for_func(func, kwargs) == expected
