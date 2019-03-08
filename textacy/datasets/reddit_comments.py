# -*- coding: utf-8 -*-
"""
Reddit Comments
---------------

A collection of up to ~1.5 billion Reddit comments posted from
October 2007 through May 2015.

Records include the following key fields (plus a few others):

    * ``body``: Full text of the comment.
    * ``created_utc``: Date on which the comment was posted.
    * ``subreddit``: Sub-reddit in which the comment was posted, excluding the
      familiar "/r/" prefix.
    * ``score``: Net score (upvotes - downvotes) on the comment.
    * ``gilded``: Number of times this comment received reddit gold.

The raw data was originally collected by /u/Stuck_In_the_Matrix via Reddit's
APIS, and stored for posterity by the `Internet Archive <https://archive.org>`_.
For more details, refer to https://archive.org/details/2015_reddit_comments_corpus.
"""
from __future__ import unicode_literals

import logging
import os
import re
from datetime import datetime

import requests

from .. import compat
from .. import data_dir
from .. import io
from .. import preprocess
from .base import Dataset

LOGGER = logging.getLogger(__name__)

NAME = "reddit_comments"
DESCRIPTION = (
    "Collection of ~1.5 billion publicly available Reddit comments "
    "from October 2007 through May 2015."
)
SITE_URL = "https://archive.org/details/2015_reddit_comments_corpus"
DATA_DIR = os.path.join(data_dir, NAME)

DOWNLOAD_ROOT = "https://archive.org/download/2015_reddit_comments_corpus/reddit_data/"
MIN_SCORE = -2147483647
MAX_SCORE = 2147483647

REDDIT_LINK_RE = re.compile(r"\[([^]]+)\]\(https?://[^\)]+\)")


class RedditComments(Dataset):
    """
    Stream Reddit comments from 1 or more compressed files on disk, either
    as texts (str) or records (dict) with both text content and metadata.

    Download 1 or more files, optionally within a given date range::

        >>> rc = RedditComments()
        >>> rc.download(date_range=('2007-10', '2008-01'))
        >>> rc.info
        {'data_dir': 'path/to/textacy/data/reddit_comments',
         'description': 'An archive of ~1.5 billion publicly available Reddit comments from October 2007 through May 2015.',
         'name': 'reddit_comments',
         'site_url': 'https://archive.org/details/2015_reddit_comments_corpus'}

    Iterate over comments as plain texts or records with both text and metadata::

        >>> for text in rc.texts(limit=5):
        ...     print(text)
        >>> for record in rc.records(limit=5):
        ...     print(record['body'], record['author'], record['created_utc'])

    Filter comments by a variety of metadata fields and text length::

        >>> for record in rc.records(subreddit='politics', limit=5):
        ...     print(record['body'], record['score'])
        >>> for record in rc.records(date_range=('2015-01-01', '2015-01-02'), limit=5):
        ...     print(record['created_utc'])
        >>> for record in rc.records(min_len=1000, limit=1):
        ...     print(record['body'], len(record['body']))

    Args:
        data_dir (str): Path to directory on disk under which the data is stored.
            Each file covers a given month, as indicated in the filenames like
            "YYYY/RC_YYYY-MM.bz2".

    Attributes:
        min_date (str): Earliest date for which comments are available, as an
            ISO-formatted string ("YYYY-MM-DD").
        max_date (str): Latest date for which comments are available, as an
            ISO-formatted string ("YYYY-MM-DD").
        filenames (Tuple[str]): Full paths on disk for all Reddit comments files
            found under :attr:`ReddictComments.data_dir` directory, sorted
            in chronological order.
    """

    min_date = "2007-10-01"
    max_date = "2015-06-01"

    def __init__(self, data_dir=DATA_DIR):
        super(RedditComments, self).__init__(
            name=NAME, description=DESCRIPTION, site_url=SITE_URL, data_dir=data_dir
        )

    @property
    def filenames(self):
        """
        Tuple[str]: Full paths on disk for all Reddit comments files found under
            the ``data_dir`` directory, sorted chronologically.
        """
        if os.path.exists(self.data_dir):
            return tuple(
                sorted(
                    io.get_filenames(self.data_dir, extension=".bz2", recursive=True)
                )
            )
        else:
            LOGGER.warning("%s data directory does not exist", self.data_dir)
            return tuple()

    def download(self, date_range=(None, None), force=False):
        """
        Download 1 or more monthly Reddit comments files from archive.org
        and save them to disk under the ``data_dir`` used to instantiate.

        Args:
            date_range (Tuple[str]): Interval specifying the [start, end) dates
                for which comments files will be downloaded. Each item must be
                a str formatted as YYYY-MM or YYYY-MM-DD (the latter is converted
                to the corresponding YYYY-MM value). Both start and end values
                must be specified, but a null value for either is automatically
                replaced by the minimum or maximum valid values, respectively.
            force (bool): If True, download the dataset, even if it already
                exists on disk under ``data_dir``.
        """
        date_range = self._parse_date_range(date_range)
        fstubs = self._generate_filestubs(date_range)
        for fstub in fstubs:
            url = compat.urljoin(DOWNLOAD_ROOT, fstub)
            filename = os.path.join(self.data_dir, fstub)
            if os.path.isfile(filename) and force is False:
                LOGGER.warning("File %s already exists; skipping download...", filename)
                continue
            LOGGER.info("Downloading data from %s and writing it to %s", url, filename)
            io.write_http_stream(
                url, filename, mode="wb", encoding=None, make_dirs=True, chunk_size=1024
            )

    def _parse_score_range(self, score_range):
        """Flexibly parse score range args."""
        if not isinstance(score_range, (list, tuple)):
            raise ValueError(
                "`score_range` must be a list or tuple, not {}".format(
                    type(score_range)
                )
            )
        if len(score_range) != 2:
            raise ValueError("`score_range` must have exactly two items: min and max")
        if not score_range[0]:
            score_range = (MIN_SCORE, score_range[1])
        if not score_range[1]:
            score_range = (score_range[0], MAX_SCORE)
        return tuple(score_range)

    def _generate_filestubs(self, date_range):
        """
        Generate a list of monthly filenames in the interval [start, end),
        each with format "YYYY/RC_YYYY-MM.bz2".
        """
        fstubs = []
        yrmo, end = date_range
        while yrmo < end:
            # parse current yrmo
            try:
                dt = datetime.strptime(yrmo, "%Y-%m")
            except ValueError:
                dt = datetime.strptime(yrmo, "%Y-%m-%d")
            fstubs.append(dt.strftime("%Y/RC_%Y-%m.bz2"))
            # dead simple iteration to next yrmo
            next_yr = dt.year
            next_mo = dt.month + 1
            if next_mo > 12:
                next_yr += 1
                next_mo = 1
            yrmo = datetime(next_yr, next_mo, 1).strftime("%Y-%m")
        return tuple(fstubs)

    def texts(
        self, subreddit=None, date_range=None, score_range=None, min_len=0, limit=-1
    ):
        """
        Iterate over comments (text-only) in 1 or more files of this dataset,
        optionally filtering by a variety of metadata and/or text length,
        in chronological order.

        Args:
            subreddit (str or Set[str]): Filter comments for those which were
                posted in the specified subreddit(s).
            date_range (Tuple[str]): Filter comments for those which were posted
                within the interval [start, end). Each item must be a str in
                ISO-standard format, i.e. some amount of YYYY-MM-DDTHH:mm:ss.
                Both start and end values must be specified, but a null value
                for either is automatically replaced by the minimum or maximum
                valid values, respectively.
            score_range (Tuple[int]): Filter comments for those whose score
                (# upvotes minus # downvotes) is within the interval [low, high).
                Both start and end values must be specified, but a null value
                for either is automatically replaced by the minimum or maximum
                valid values, respectively.
            min_len (int): Filter comments for those whose body length in chars
                is at least this long.
            limit (int): Maximum number of comments passing all filters to yield.
                If -1, all comments are iterated over.

        Yields:
            str: Plain text of next (by chronological order) comment in dataset
            passing all filter params.
        """
        texts = self._iterate(
            True,
            subreddit=subreddit,
            date_range=date_range,
            score_range=score_range,
            min_len=min_len,
            limit=limit,
        )
        for text in texts:
            yield text

    def records(
        self, subreddit=None, date_range=None, score_range=None, min_len=0, limit=-1
    ):
        """
        Iterate over comments (including text and metadata) in 1 or more files
        of this dataset, optionally filtering by a variety of metadata and/or
        text length, in chronological order.

        Args:
            subreddit (str or Set[str]): Filter comments for those which were
                posted in the specified subreddit(s).
            date_range (Tuple[str]): Filter comments for those which were posted
                within the interval [start, end). Each item must be a str in
                ISO-standard format, i.e. some amount of YYYY-MM-DDTHH:mm:ss.
                Both start and end values must be specified, but a null value
                for either is automatically replaced by the minimum or maximum
                valid values, respectively.
            score_range (Tuple[int]): Filter comments for those whose score
                (# upvotes minus # downvotes) is within the interval [low, high).
                Both start and end values must be specified, but a null value
                for either is automatically replaced by the minimum or maximum
                valid values, respectively.
            min_len (int): Filter comments for those whose body length in chars
                is at least this long.
            limit (int): Maximum number of comments passing all filters to yield.
                If -1, all comments are iterated over.

        Yields:
            dict: Full text and metadata of next (by chronological order) comment
            in dataset passing all filter params.
        """
        records = self._iterate(
            False,
            subreddit=subreddit,
            date_range=date_range,
            score_range=score_range,
            min_len=min_len,
            limit=limit,
        )
        for record in records:
            yield record

    def _iterate(self, text_only, subreddit, date_range, score_range, min_len, limit):
        """
        Low-level method to iterate over the records in this dataset. Used by
        :meth:`RedditComments.texts()` and :meth:`RedditComments.records()`.
        """
        if subreddit:
            if isinstance(subreddit, compat.string_types):
                subreddit = {subreddit}
            elif isinstance(subreddit, (list, tuple)):
                subreddit = set(subreddit)
        if score_range:
            score_range = self._parse_score_range(score_range)
        if date_range:
            date_range = self._parse_date_range(date_range)
            needed_filenames = {
                os.path.join(self.data_dir, fstub)
                for fstub in self._generate_filestubs(date_range)
            }
            filenames = tuple(
                fname for fname in self.filenames if fname in needed_filenames
            )
        else:
            filenames = self.filenames

        if not filenames:
            raise IOError(
                "No files found at {} corresponding to date range {}".format(
                    self.data_dir, date_range
                )
            )

        n = 0
        for filename in filenames:
            for line in io.read_json(filename, mode="rb", lines=True):

                if subreddit and line["subreddit"] not in subreddit:
                    continue
                if score_range and not score_range[0] <= line["score"] < score_range[1]:
                    continue
                line["created_utc"] = self._convert_timestamp(
                    line.get("created_utc", "")
                )
                if (
                    date_range
                    and not date_range[0] <= line["created_utc"] < date_range[1]
                ):
                    continue
                line["body"] = self._clean_content(line["body"])
                if min_len and len(line["body"]) < min_len:
                    continue

                if text_only is True:
                    yield line["body"]
                else:
                    line["retrieved_on"] = self._convert_timestamp(
                        line.get("retrieved_on", "")
                    )
                    yield line

                n += 1
                if n == limit:
                    break

            if n == limit:
                break

    def _convert_timestamp(self, timestamp):
        try:
            return datetime.utcfromtimestamp(int(timestamp)).strftime(
                "%Y-%m-%dT%H:%M:%S"
            )
        except (ValueError, TypeError):
            return ""

    def _clean_content(self, content):
        # strip out link markup, e.g. [foo](http://foo.com)
        content = REDDIT_LINK_RE.sub(r"\1", content)
        # clean up basic HTML cruft
        content = content.replace("&gt;", ">").replace("&lt;", "<")
        # strip out text markup, e.g. * for bold text
        content = content.replace("`", "").replace("*", "").replace("~", "")
        # normalize whitespace
        return preprocess.normalize_whitespace(content)
