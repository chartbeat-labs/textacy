from __future__ import absolute_import, unicode_literals

import pytest

from textacy import compat
from textacy.datasets import dataset


DATASET = dataset.Dataset("foo", {"test": True})


def test_repr():
    assert compat.to_unicode(str(DATASET)).startswith("Dataset")


def test_info():
    assert isinstance(DATASET.info, dict)
    assert "name" in DATASET.info
    assert "test" in DATASET.info


def test_iter():
    assert isinstance(DATASET, compat.Iterable)


def test_methods():
    for name in ("texts", "records", "download"):
        assert hasattr(DATASET, name)
        with pytest.raises(NotImplementedError):
            getattr(DATASET, name)()
