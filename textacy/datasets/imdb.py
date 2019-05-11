# -*- coding: utf-8 -*-
"""
IMDB Reviews
------------

A collection of 50k highly polar movie reviews posted to IMDB, split evenly
into training and testing sets, with 25k positive and 25k negative sentiment labels,
as well as some unlabeled reviews.

Records include the following key fields (plus a few others):

    * ``text``: Full text of the review.
    * ``subset``: Subset of the dataset ("train" or "test") into which
      the review has been split.
    * ``label``: Sentiment label ("pos" or "neg") assigned to the review.
    * ``rating``: Numeric rating assigned by the original reviewer, ranging from
      1 to 10. Reviews with a rating <= 5 are "neg"; the rest are "pos".
    * ``movie_id``: Unique identifier for the movie under review within IMDB,
      useful for grouping reviews or joining with an external movie dataset.

Reference: Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng,
and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis.
The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import io
import itertools
import logging
import os
import re

from .. import compat
from .. import constants
from .. import io as tio
from . import utils
from .dataset import Dataset

LOGGER = logging.getLogger(__name__)

NAME = "imdb"
META = {
    "site_url": "http://ai.stanford.edu/~amaas/data/sentiment",
    "description": (
        "Collection of 50k highly polar movie reviews split evenly "
        "into train and test sets, with 25k positive and 25k negative labels. "
        "Also includes some unlabeled reviews."
    )
}
DOWNLOAD_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

RE_MOVIE_ID = re.compile(r"/(tt\d+)/")


class IMDB(Dataset):
    """
    Stream a collection of IMDB movie reviews from text files on disk,
    either as texts or text + metadata pairs.

    Download the data (one time only!), saving and extracting its contents to disk::

        >>> imdb = IMDB()
        >>> imdb.download()
        >>> imdb.info
        {'name': 'imdb',
         'site_url': 'http://ai.stanford.edu/~amaas/data/sentiment',
         'description': 'Collection of 50k highly polar movie reviews split evenly into train and test sets, with 25k positive and 25k negative labels. Also includes some unlabeled reviews.'}

    Iterate over movie reviews as texts or records with both text and metadata::

        >>> for text in ds.texts(limit=5):
        ...     print(text)
        >>> for text, meta in ds.records(limit=5):
        ...     print("\\n{} {}\\n{}".format(meta["label"], meta["rating"], text))

    Filter movie reviews by a variety of metadata fields and text length::

        >>> for text, meta in ds.records(label="pos", limit=5):
        ...     print(meta["rating"], ":", text)
        >>> for text, meta in ds.records(rating_range=(9, 11), limit=5):
        ...     print(meta["rating"], text)
        >>> for text in ds.texts(min_len=1000, limit=5):
        ...     print(len(text))

    Stream movie reviews into a :class:`textacy.Corpus`::

        >>> textacy.Corpus("en", data=ds.records(limit=100))
        Corpus(100 docs; 24340 tokens)

    Args:
        data_dir (str): Path to directory on disk under which the data is stored.

    Attributes:
        full_rating_range (Tuple[int]): Lowest and highest ratings for which
            movie reviews are available.
    """

    full_rating_range = (1, 10)

    def __init__(self, data_dir=os.path.join(constants.DEFAULT_DATA_DIR, NAME)):
        super(IMDB, self).__init__(NAME, meta=META)
        self.data_dir = data_dir
        self._movie_ids = {"train": {}, "test": {}}
        self._subset_labels = {
            "train": ("pos", "neg", "unsup"),
            "test": ("pos", "neg"),
        }
        self._subset = None
        self._label = None

    def download(self, force=False):
        """
        Download the data as a compressed tar archive file, then save it to disk and
        extract its contents under the ``data_dir`` directory.

        Args:
            force (bool): If True, always download the dataset even if
                it already exists.
        """
        filepath = utils.download_file(
            DOWNLOAD_URL,
            filename="aclImdb.tar.gz",
            dirpath=self.data_dir,
            force=force,
        )
        if filepath:
            utils.unpack_archive(filepath, extract_dir=None)
        self._check_data()

    def _check_data(self):
        """Check that necessary data is found on disk, or raise an OSError."""
        data_dirpaths = (
            os.path.join(self.data_dir, "aclImdb", subset, label)
            for subset, labels in self._subset_labels.items()
            for label in labels
        )
        url_filepaths = (
            os.path.join(self.data_dir, "aclImdb", subset, "urls_{}.txt".format(label))
            for subset, labels in self._subset_labels.items()
            for label in labels
        )
        for dirpath in data_dirpaths:
            if not os.path.isdir(dirpath):
                raise OSError(
                    "data directory {} not found; "
                    "has the dataset been downloaded?".format(dirpath)
                )
        for filepath in url_filepaths:
            if not os.path.isfile(filepath):
                raise OSError(
                    "data file {} not found; "
                    "has the dataset been downloaded?".format(filepath)
                )

    def __iter__(self):
        self._check_data()
        dirpaths = tuple(
            os.path.join(self.data_dir, "aclImdb", subset, label)
            for subset in self._subset or self._subset_labels.keys()
            for label in self._label or self._subset_labels[subset]
        )
        for dirpath in dirpaths:
            for filepath in tio.get_filepaths(dirpath, match_regex=r"^\d+_\d+\.txt$"):
                yield self._load_record(filepath)

    def _load_record(self, filepath):
        dirpath, filename = os.path.split(filepath)
        dirpath, label = os.path.split(dirpath)
        _, subset = os.path.split(dirpath)
        id_, rating = filename[:-4].split("_")
        with io.open(filepath, mode="rt", encoding="utf-8") as f:
            text = f.read().replace("<br />", "\n").strip()
        return {
            "text": text,
            "subset": subset,
            "label": label,
            "rating": int(rating) if label != "unsup" else None,
            "movie_id": self._get_movie_id(subset, label, int(id_))
        }

    def _get_movie_id(self, subset, label, id_):
        try:
            return self._movie_ids[subset][label][id_]
        except KeyError:
            fpath = os.path.join(
                self.data_dir, "aclImdb", subset, "urls_{}.txt".format(label))
            self._movie_ids[subset][label] = {
                id_: RE_MOVIE_ID.search(line).group(1)
                for id_, line in enumerate(tio.read_text(fpath, mode="rt", lines=True))
            }
            return self._movie_ids[subset][label][id_]

    def _get_filters(self, rating_range, min_len):
        filters = []
        if min_len is not None:
            if min_len < 1:
                raise ValueError("`min_len` must be at least 1")
            filters.append(
                lambda record: len(record.get("text", "")) >= min_len
            )
        if rating_range is not None:
            date_range = utils.validate_and_clip_range_filter(
                rating_range, self.full_rating_range, val_type=int)
            filters.append(
                lambda record: (
                    record.get("rating")
                    and rating_range[0] <= record["rating"] < rating_range[1]
                )
            )
        return filters

    def _filtered_iter(self, filters):
        if filters:
            for record in self:
                if all(filter_(record) for filter_ in filters):
                    yield record
        else:
            for record in self:
                yield record

    def texts(self, subset=None, label=None, rating_range=None, min_len=None, limit=None):
        """
        Iterate over movie reviews in this dataset, optionally filtering by
        a variety of metadata and/or text length, and yield texts only.

        Args:
            subset (str, {"train", "test"}): Filter movie reviews by
                the dataset subset into which they've already been split.
            label (str, {"pos", "neg", "unsup"}): Filter movie reviews by
                the assigned sentiment label (or lack thereof, for "unsup").
            min_len (int): Filter movie reviews by the length (number of characters)
                of their text content.
            limit (int): Yield no more than ``limit`` movie reviews that match all
                specified filters.

        Yields:
            str: Text of the next movie review in dataset passing all filters.
        """
        self._subset = utils.to_collection(subset, compat.string_types, tuple)
        self._label = utils.to_collection(label, compat.string_types, tuple)
        try:
            filters = self._get_filters(rating_range, min_len)
            for record in itertools.islice(self._filtered_iter(filters), limit):
                yield record["text"]
        finally:
            self._subset = None
            self._label = None

    def records(self, subset=None, label=None, rating_range=None, min_len=None, limit=None):
        """
        Iterate over movie reviews in this dataset, optionally filtering by
        a variety of metadata and/or text length, and yield text + metadata pairs.

        Args:
            subset (str, {"train", "test"}): Filter movie reviews by
                the dataset subset into which they've already been split.
            label (str, {"pos", "neg", "unsup"}): Filter movie reviews by
                the assigned sentiment label (or lack thereof, for "unsup").
            min_len (int): Filter movie reviews by the length (number of characters)
                of their text content.
            limit (int): Yield no more than ``limit`` movie reviews that match all
                specified filters.

        Yields:
            str: Text of the next movie review in dataset passing all filters.
            dict: Metadata of the next movie review in dataset passing all filters.
        """
        self._subset = utils.to_collection(subset, compat.string_types, tuple)
        self._label = utils.to_collection(label, compat.string_types, tuple)
        try:
            filters = self._get_filters(rating_range, min_len)
            for record in itertools.islice(self._filtered_iter(filters), limit):
                yield record.pop("text"), record
        finally:
            self._subset = None
            self._label = None
