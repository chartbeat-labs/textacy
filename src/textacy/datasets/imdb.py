"""
IMDB movie reviews
------------------

A collection of 50k highly polar movie reviews posted to IMDB, split evenly
into training and testing sets, with 25k positive and 25k negative sentiment labels,
as well as some unlabeled reviews.

Records include the following key fields (plus a few others):

    - ``text``: Full text of the review.
    - ``subset``: Subset of the dataset ("train" or "test") into which
      the review has been split.
    - ``label``: Sentiment label ("pos" or "neg") assigned to the review.
    - ``rating``: Numeric rating assigned by the original reviewer, ranging from
      1 to 10. Reviews with a rating <= 5 are "neg"; the rest are "pos".
    - ``movie_id``: Unique identifier for the movie under review within IMDB,
      useful for grouping reviews or joining with an external movie dataset.

Reference: Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng,
and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis.
The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).
"""
from __future__ import annotations

import io
import itertools
import logging
import os
import re
from typing import Any, ClassVar, Iterable, Optional

from .. import constants
from .. import io as tio
from .. import types, utils
from .base import Dataset


LOGGER = logging.getLogger(__name__)

NAME = "imdb"
META = {
    "site_url": "http://ai.stanford.edu/~amaas/data/sentiment",
    "description": (
        "Collection of 50k highly polar movie reviews split evenly "
        "into train and test sets, with 25k positive and 25k negative labels. "
        "Also includes some unlabeled reviews."
    ),
}
DOWNLOAD_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

RE_MOVIE_ID = re.compile(r"/(tt\d+)/")


class IMDB(Dataset):
    """
    Stream a collection of IMDB movie reviews from text files on disk,
    either as texts or text + metadata pairs.

    Download the data (one time only!), saving and extracting its contents to disk::

        >>> import textacy.datasets
        >>> ds = textacy.datasets.IMDB()
        >>> ds.download()
        >>> ds.info
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

    Stream movie reviews into a :class:`textacy.Corpus <textacy.corpus.Corpus>`::

        >>> textacy.Corpus("en", data=ds.records(limit=100))
        Corpus(100 docs; 24340 tokens)

    Args:
        data_dir: Path to directory on disk under which the data is stored,
            i.e. ``/path/to/data_dir/imdb``.

    Attributes:
        full_rating_range: Lowest and highest ratings for which movie reviews are available.
    """

    full_rating_range: ClassVar[tuple[int, int]] = (1, 10)

    def __init__(
        self,
        data_dir: types.PathLike = constants.DEFAULT_DATA_DIR.joinpath(NAME),
    ):
        super().__init__(NAME, meta=META)
        self.data_dir = utils.to_path(data_dir).resolve()
        self._movie_ids: dict[str, dict] = {"train": {}, "test": {}}
        self._subset_labels: dict[str, tuple[str, ...]] = {
            "train": ("pos", "neg", "unsup"),
            "test": ("pos", "neg"),
        }
        self._subset: Optional[tuple[str, ...]] = None
        self._label: Optional[tuple[str, ...]] = None

    def download(self, *, force: bool = False) -> None:
        """
        Download the data as a compressed tar archive file, then save it to disk and
        extract its contents under the ``data_dir`` directory.

        Args:
            force: If True, always download the dataset even if it already exists
                on disk under ``data_dir``.
        """
        filepath = tio.download_file(
            DOWNLOAD_URL, filename="aclImdb.tar.gz", dirpath=self.data_dir, force=force
        )
        if filepath:
            tio.unpack_archive(filepath, extract_dir=None)
        self._check_data()

    def _check_data(self):
        """Check that necessary data is found on disk, or raise an OSError."""
        data_dirpaths = (
            self.data_dir.joinpath("aclImdb", subset, label)
            for subset, labels in self._subset_labels.items()
            for label in labels
        )
        url_filepaths = (
            self.data_dir.joinpath("aclImdb", subset, f"urls_{label}.txt")
            for subset, labels in self._subset_labels.items()
            for label in labels
        )
        for dirpath in data_dirpaths:
            if not dirpath.is_dir():
                raise OSError(
                    f"data directory {dirpath} not found; "
                    "has the dataset been downloaded?"
                )
        for filepath in url_filepaths:
            if not filepath.is_file():
                raise OSError(
                    f"data file {filepath} not found; has the dataset been downloaded?"
                )

    def __iter__(self):
        self._check_data()
        dirpaths = tuple(
            self.data_dir.joinpath("aclImdb", subset, label)
            for subset in self._subset or self._subset_labels.keys()
            for label in self._label or self._subset_labels[subset]
        )
        for dirpath in dirpaths:
            for filepath in tio.get_filepaths(dirpath, match_regex=r"^\d+_\d+\.txt$"):
                yield self._load_record(filepath)

    def _load_record(self, filepath: str) -> dict[str, Any]:
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
            "movie_id": self._get_movie_id(subset, label, int(id_)),
        }

    def _get_movie_id(self, subset, label, id_):
        try:
            return self._movie_ids[subset][label][id_]
        except KeyError:
            fpath = self.data_dir.joinpath("aclImdb", subset, f"urls_{label}.txt")
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
            filters.append(lambda record: len(record.get("text", "")) >= min_len)
        if rating_range is not None:
            rating_range = utils.validate_and_clip_range(
                rating_range, self.full_rating_range, val_type=int
            )
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

    def texts(
        self,
        *,
        subset: Optional[str] = None,
        label: Optional[str] = None,
        rating_range: Optional[tuple[Optional[int], Optional[int]]] = None,
        min_len: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Iterable[str]:
        """
        Iterate over movie reviews in this dataset, optionally filtering by
        a variety of metadata and/or text length, and yield texts only.

        Args:
            subset ({"train", "test"}): Filter movie reviews
                by the dataset subset into which they've already been split.
            label ({"pos", "neg", "unsup"}): Filter movie reviews
                by the assigned sentiment label (or lack thereof, for "unsup").
            rating_range: Filter movie reviews by the rating assigned by the reviewer.
                Only those with ratings in the interval [low, high) are included.
                Both low and high values must be specified, but a null value
                for either is automatically replaced by the minimum or maximum
                valid values, respectively.
            min_len: Filter reviews by the length (# characters) of their text content.
            limit: Yield no more than ``limit`` reviews that match all specified filters.

        Yields:
            Text of the next movie review in dataset passing all filters.

        Raises:
            ValueError: If any filtering options are invalid.
        """
        self._subset = utils.to_tuple(subset) if subset is not None else None
        self._label = utils.to_tuple(label) if label is not None else None
        try:
            filters = self._get_filters(rating_range, min_len)
            for record in itertools.islice(self._filtered_iter(filters), limit):
                yield record["text"]
        finally:
            self._subset = None
            self._label = None

    def records(
        self,
        *,
        subset: Optional[str] = None,
        label: Optional[str] = None,
        rating_range: Optional[tuple[Optional[int], Optional[int]]] = None,
        min_len: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Iterable[types.Record]:
        """
        Iterate over movie reviews in this dataset, optionally filtering by
        a variety of metadata and/or text length, and yield text + metadata pairs.

        Args:
            subset ({"train", "test"}): Filter movie reviews
                by the dataset subset into which they've already been split.
            label ({"pos", "neg", "unsup"}): Filter movie reviews
                by the assigned sentiment label (or lack thereof, for "unsup").
            rating_range: Filter movie reviews by the rating assigned by the reviewer.
                Only those with ratings in the interval [low, high) are included.
                Both low and high values must be specified, but a null value
                for either is automatically replaced by the minimum or maximum
                valid values, respectively.
            min_len: Filter reviews by the length (# characters) of their text content.
            limit: Yield no more than ``limit`` reviews that match all specified filters.

        Yields:
            Text of the next movie review in dataset passing all filters,
            and its corresponding metadata.

        Raises:
            ValueError: If any filtering options are invalid.
        """
        self._subset = utils.to_tuple(subset) if subset is not None else None
        self._label = utils.to_tuple(label) if label is not None else None
        try:
            filters = self._get_filters(rating_range, min_len)
            for record in itertools.islice(self._filtered_iter(filters), limit):
                yield types.Record(text=record.pop("text"), meta=record)
        finally:
            self._subset = None
            self._label = None
