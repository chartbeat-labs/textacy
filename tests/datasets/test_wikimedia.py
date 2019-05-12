# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

import os
import re

import pytest

from textacy import compat
from textacy.datasets import wikimedia


WIKINEWS = wikimedia.Wikinews(lang="en", version="current")
WIKIPEDIA = wikimedia.Wikipedia(lang="en", version="current")


@pytest.mark.skipif(
    WIKIPEDIA.filepath is None,
    reason="Wikinews dataset must be downloaded before running tests",
)
class TestWikipedia(object):

    @pytest.mark.skip("No need to download a new dataset every time")
    def test_download(self, tmpdir):
        dataset = wikimedia.Wikipedia(data_dir=str(tmpdir))
        dataset.download()
        assert os.path.isfile(dataset.filepath)

    def test_oserror(self, tmpdir):
        dataset = wikimedia.Wikipedia(data_dir=str(tmpdir))
        with pytest.raises(OSError):
            _ = list(dataset.texts())

    def test_texts(self):
        texts = list(WIKIPEDIA.texts(limit=3))
        assert len(texts) > 0
        for text in texts:
            assert isinstance(text, compat.unicode_)

    def test_texts_limit(self):
        for limit in (1, 5, 10):
            assert sum(1 for _ in WIKIPEDIA.texts(limit=limit)) == limit

    def test_texts_min_len(self):
        for min_len in (100, 200, 500):
            assert all(
                len(text) >= min_len
                for text in WIKIPEDIA.texts(min_len=min_len, limit=10)
            )

    def test_records(self):
        for text, meta in WIKIPEDIA.records(limit=3):
            assert isinstance(text, compat.unicode_)
            assert isinstance(meta, dict)

    def test_records_limit(self):
        for limit in (1, 5, 10):
            assert sum(1 for _ in WIKIPEDIA.records(limit=limit)) == limit

    def test_records_min_len(self):
        for min_len in (100, 200, 500):
            assert all(
                len(text) >= min_len
                for text, meta in WIKIPEDIA.records(min_len=min_len, limit=10)
            )

    def test_records_category(self):
        categories = (
            {"Living people"},
            {"All stub articles", "Coordinates on Wikidata"},
        )
        for category in categories:
            records = list(WIKIPEDIA.records(category=category, limit=3))
            assert any(
                ctgry in meta["categories"]
                for _, meta in records
                for ctgry in category
            )

    def test_records_wiki_link(self):
        wiki_links = (
            {"United_States"},
            {"France", "England"},
        )
        for wiki_link in wiki_links:
            records = list(WIKIPEDIA.records(wiki_link=wiki_link, limit=3))
            assert any(
                wl in meta["wiki_links"]
                for _, meta in records
                for wl in wiki_link
            )


@pytest.mark.skipif(
    WIKINEWS.filepath is None,
    reason="Wikinews dataset must be downloaded before running tests",
)
class TestWikinews(object):

    @pytest.mark.skip("No need to download a new dataset every time")
    def test_download(self, tmpdir):
        dataset = wikimedia.Wikinews(data_dir=str(tmpdir))
        dataset.download()
        assert os.path.isfile(dataset.filepath)

    def test_oserror(self, tmpdir):
        dataset = wikimedia.Wikinews(data_dir=str(tmpdir))
        with pytest.raises(OSError):
            _ = list(dataset.texts())

    def test_texts(self):
        texts = list(WIKINEWS.texts(limit=3))
        assert len(texts) > 0
        for text in texts:
            assert isinstance(text, compat.unicode_)

    def test_texts_limit(self):
        for limit in (1, 5, 10):
            assert sum(1 for _ in WIKINEWS.texts(limit=limit)) == limit

    def test_texts_min_len(self):
        for min_len in (100, 200, 500):
            assert all(
                len(text) >= min_len
                for text in WIKINEWS.texts(min_len=min_len, limit=10)
            )

    def test_records(self):
        for text, meta in WIKINEWS.records(limit=3):
            assert isinstance(text, compat.unicode_)
            assert isinstance(meta, dict)

    def test_records_limit(self):
        for limit in (1, 5, 10):
            assert sum(1 for _ in WIKINEWS.records(limit=limit)) == limit

    def test_records_min_len(self):
        for min_len in (100, 200, 500):
            assert all(
                len(text) >= min_len
                for text, meta in WIKINEWS.records(min_len=min_len, limit=10)
            )

    def test_records_category(self):
        categories = (
            {"Politics and conflicts"},
            {"United States", "North America"},
        )
        for category in categories:
            records = list(WIKINEWS.records(category=category, limit=3))
            assert any(
                ctgry in meta["categories"]
                for _, meta in records
                for ctgry in category
            )

    def test_records_wiki_link(self):
        wiki_links = (
            {"Reuters"},
            {"United_States", "BBC_News"},
        )
        for wiki_link in wiki_links:
            records = list(WIKINEWS.records(wiki_link=wiki_link, limit=3))
            assert any(
                wl in meta["wiki_links"]
                for _, meta in records
                for wl in wiki_link
            )
