# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

import pytest

from textacy import compat


STRINGS = (b"bytes", "unicode", "úñîçødé")
NOT_STRINGS = (1, 2.0, ["foo", "bar"], {"foo": "bar"})


def test_to_unicode():
    for obj in STRINGS:
        assert isinstance(compat.to_unicode(obj), compat.unicode_)
    for obj in NOT_STRINGS:
        with pytest.raises(TypeError):
            compat.to_unicode(obj)


def test_to_bytes():
    for obj in STRINGS:
        assert isinstance(compat.to_bytes(obj), bytes)
    for obj in NOT_STRINGS:
        with pytest.raises(TypeError):
            compat.to_bytes(obj)


def test_builtin_iterators():
    assert not isinstance(compat.zip_([1, 2], ["a", "b"]), list)
    assert not isinstance(compat.range_(10), list)


def test_string_types():
    for s in STRINGS:
        assert isinstance(s, compat.string_types)


def test_url_quoting():
    unquoted_url = "https://github.com/chartbeat-labs/textacy"
    quoted_url = "https%3A%2F%2Fgithub.com%2Fchartbeat-labs%2Ftextacy"
    assert compat.url_unquote_plus(unquoted_url) == unquoted_url
    assert compat.url_unquote_plus(quoted_url) == unquoted_url
