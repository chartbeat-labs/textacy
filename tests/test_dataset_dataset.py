from __future__ import absolute_import, unicode_literals

import datetime
import os

import pytest

from textacy import compat
from textacy.datasets.dataset import (
    Dataset,
    validate_and_clip_range,
    _download,
    _get_filename_from_url,
)


class TestValidateAndClipRange(object):

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
            output = validate_and_clip_range(*input_)
            assert isinstance(output, tuple)
            assert len(output) == 2
            assert output[0] == max(input_[0][0], input_[1][0])
            assert output[1] == min(input_[0][1], input_[1][1])

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
                validate_and_clip_range(*input_)

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
                validate_and_clip_range(*input_)


def test_get_filename_from_url():
    url_fnames = [
        ["http://www.foo.bar/bat.zip", "bat.zip"],
        ["www.foo.bar/bat.tar.gz", "bat.tar.gz"],
        ["foo.bar/bat.zip?q=test", "bat.zip"],
        ["http%3A%2F%2Fwww.foo.bar%2Fbat.tar.gz", "bat.tar.gz"]
    ]
    for url, fname in url_fnames:
        assert _get_filename_from_url(url) == fname
