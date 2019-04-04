# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

import pytest

from textacy import compat


strings = (b"bytes", "unicode", "úñîçødé")
not_strings = (1, 2.0, ["foo", "bar"], {"foo": "bar"})


def test_to_unicode():
    for obj in strings:
        assert isinstance(compat.to_unicode(obj), compat.unicode_)
    for obj in not_strings:
        with pytest.raises(TypeError):
            compat.to_unicode(obj)


def test_to_bytes():
    for obj in strings:
        assert isinstance(compat.to_bytes(obj), bytes)
    for obj in not_strings:
        with pytest.raises(TypeError):
            compat.to_bytes(obj)
