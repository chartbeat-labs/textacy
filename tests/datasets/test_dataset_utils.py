from __future__ import absolute_import, unicode_literals

import datetime
import os
import tarfile
import zipfile

import pytest

from textacy import compat
from textacy.datasets.utils import (
    validate_and_clip_range_filter,
    validate_set_member_filter,
    download_file,
    get_filename_from_url,
    to_collection,
    unpack_archive,
)
from textacy import io as tio


class TestValidateAndClipRangeFilter(object):

    def test_good_inputs(self):
        inputs = [
            [("2001-01", "2002-01"), ("2000-01", "2003-01")],
            [["2001-01", "2004-01"], ("2000-01", "2003-01")],
            [("2001-01", "2002-01"), ["2000-01", "2003-01"], compat.string_types],
            [[-5, 5], [-10, 10]],
            [(-5, 5), (0, 10)],
            [(-5, 5), (-10, 10), int],
            [(-5, 5), (-10, 10), (int, float)],
        ]
        for input_ in inputs:
            output = validate_and_clip_range_filter(*input_)
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
            output = validate_and_clip_range_filter(*input_)
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
            [[-5, 5], [-10, 10], compat.string_types],
        ]
        for input_ in inputs:
            with pytest.raises(TypeError):
                validate_and_clip_range_filter(*input_)

    def test_bad_valueerror(self):
        inputs = [
            [("2001-01", "2002-01", "2003-01"), ("2000-01", "2003-01")],
            [("2001-01", "2002-01"), ["2000-01", "2002-01", "2004-01"]],
            [[0, 5, 10], (-10, 10)],
            [(-5, 5), [-10, 0, 10]],
            [(-5, 5), [-10, 0, 10], compat.string_types],
        ]
        for input_ in inputs:
            with pytest.raises(ValueError):
                validate_and_clip_range_filter(*input_)


class TestValidateSetMemberFilter(object):

    def test_good_inputs(self):
        inputs = [
            [{"a", "b"}, compat.string_types, {"a", "b", "c"}],
            ["a", compat.string_types, {"a", "b", "c"}],
            [("a", "b"), compat.string_types, {"a", "b", "c"}],
            [["a", "b"], compat.string_types],
            [{1, 2}, int, {1, 2, 3}],
            [{1, 2}, (int, float), {1, 2, 3}],
            [1, int, {1: "a", 2: "b", 3: "c"}],
            [{3.14, 42.0}, float],
            [3.14, (int, float)],
        ]
        for input_ in inputs:
            output = validate_set_member_filter(*input_)
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
                validate_set_member_filter(*input_)

    def test_bad_valueerror(self):
        inputs = [
            [{"a", "b"}, compat.string_types, {"x", "y", "z"}],
            [{"a", "x"}, compat.string_types, {"x", "y", "z"}],
            ["a", compat.string_types, {"x", "y", "z"}],
            ["a", compat.string_types, {"x": 24, "y": 25, "z": 26}],
        ]
        for input_ in inputs:
            with pytest.raises(ValueError):
                validate_set_member_filter(*input_)


def test_get_filename_from_url():
    url_fnames = [
        ["http://www.foo.bar/bat.zip", "bat.zip"],
        ["www.foo.bar/bat.tar.gz", "bat.tar.gz"],
        ["foo.bar/bat.zip?q=test", "bat.zip"],
        ["http%3A%2F%2Fwww.foo.bar%2Fbat.tar.gz", "bat.tar.gz"]
    ]
    for url, fname in url_fnames:
        assert get_filename_from_url(url) == fname


@pytest.mark.skipif(
    compat.PY2 is True,
    reason="PY2's stdlib complains about str/unicode/path TypeErrors. So, fuck it."
)
def test_unpack_archive(tmpdir):
    data = "Here's some text data to pack and unpack."
    fpath_txt = str(tmpdir.join("test_unpack_archive.txt"))
    with tio.open_sesame(fpath_txt, mode="wt") as f:
        f.write(data)
    fpath_zip = str(tmpdir.join("test_unpack_archive.zip"))
    with zipfile.ZipFile(fpath_zip, "w") as f:
        f.write(fpath_txt)
    unpack_archive(fpath_zip, extract_dir=tmpdir)
    fpath_tar = str(tmpdir.join("test_unpack_archive.tar"))
    with tarfile.TarFile(fpath_tar, "w") as f:
        f.add(fpath_txt)
    unpack_archive(fpath_tar, extract_dir=tmpdir)
    unpack_archive(fpath_txt, extract_dir=tmpdir)


def test_to_collection():
    in_outs = [
        [(1, int, list), [1]],
        [([1, 2], int, tuple), (1, 2)],
    ]
    assert to_collection(None, int, list) is None
    for in_, out_ in in_outs:
        assert to_collection(*in_) == out_
