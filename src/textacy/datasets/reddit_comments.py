"""
Reddit comments
---------------

A collection of up to ~1.5 billion Reddit comments posted from
October 2007 through May 2015.

Records include the following key fields (plus a few others):

    - ``body``: Full text of the comment.
    - ``created_utc``: Date on which the comment was posted.
    - ``subreddit``: Sub-reddit in which the comment was posted, excluding the
      familiar "/r/" prefix.
    - ``score``: Net score (upvotes - downvotes) on the comment.
    - ``gilded``: Number of times this comment received reddit gold.

The raw data was originally collected by /u/Stuck_In_the_Matrix via Reddit's
APIS, and stored for posterity by the `Internet Archive <https://archive.org>`_.
For more details, refer to https://archive.org/details/2015_reddit_comments_corpus.
"""
import itertools
import logging
import pathlib
import re
import urllib.parse
from datetime import datetime
from typing import Iterable, Optional, Set, Tuple, Union

from .. import constants, types, utils
from .. import io as tio
from .base import Dataset

LOGGER = logging.getLogger(__name__)

NAME = "reddit_comments"
META = {
    "site_url": "https://archive.org/details/2015_reddit_comments_corpus",
    "description": (
        "Collection of ~1.5 billion publicly available Reddit comments "
        "from October 2007 through May 2015."
    ),
}
DOWNLOAD_ROOT = "https://archive.org/download/2015_reddit_comments_corpus/reddit_data/"

RE_REDDIT_LINK = re.compile(r"\[([^]]+)\]\(https?://[^\)]+\)")


class RedditComments(Dataset):
    """
    Stream a collection of Reddit comments from 1 or more compressed files on disk,
    either as texts or text + metadata pairs.

    Download the data (one time only!) or subsets thereof by specifying a date range::

        >>> import textacy.datasets
        >>> ds = textacy.datasets.RedditComments()
        >>> ds.download(date_range=("2007-10", "2008-03"))
        >>> ds.info
        {'name': 'reddit_comments',
         'site_url': 'https://archive.org/details/2015_reddit_comments_corpus',
         'description': 'Collection of ~1.5 billion publicly available Reddit comments from October 2007 through May 2015.'}

    Iterate over comments as texts or records with both text and metadata::

        >>> for text in ds.texts(limit=5):
        ...     print(text)
        >>> for text, meta in ds.records(limit=5):
        ...     print("\\n{} {}\\n{}".format(meta["author"], meta["created_utc"], text))

    Filter comments by a variety of metadata fields and text length::

        >>> for text, meta in ds.records(subreddit="politics", limit=5):
        ...     print(meta["score"], ":", text)
        >>> for text, meta in ds.records(date_range=("2008-01", "2008-03"), limit=5):
        ...     print(meta["created_utc"])
        >>> for text, meta in ds.records(score_range=(10, None), limit=5):
        ...     print(meta["score"], ":", text)
        >>> for text in ds.texts(min_len=2000, limit=5):
        ...     print(len(text))

    Stream comments into a :class:`textacy.Corpus <textacy.corpus.Corpus>`::

        >>> textacy.Corpus("en", data=ds.records(limit=1000))
        Corpus(1000 docs; 27582 tokens)

    Args:
        data_dir: Path to directory on disk under which the data is stored,
            i.e. ``/path/to/data_dir/reddit_comments``. Each file covers a given month,
            as indicated in the filename like "YYYY/RC_YYYY-MM.bz2".

    Attributes:
        full_date_range: First and last dates for which comments
            are available, each as an ISO-formatted string (YYYY-MM-DD).
    """

    full_date_range: Tuple[str, str] = ("2007-10-01", "2015-06-01")
    _full_score_range: Tuple[int, int] = (-2147483647, 2147483647)

    def __init__(
        self,
        data_dir: Union[str, pathlib.Path] = constants.DEFAULT_DATA_DIR.joinpath(NAME),
    ):
        super().__init__(NAME, meta=META)
        self.data_dir = utils.to_path(data_dir).resolve()
        self._date_range: Optional[Tuple[Optional[str], Optional[str]]] = None

    @property
    def filepaths(self) -> Tuple[str, ...]:
        """
        Full paths on disk for all Reddit comments files found under
        :attr:`RedditComments.data_dir` directory, sorted in chronological order.
        """
        if self.data_dir.is_dir():
            return tuple(
                sorted(
                    tio.get_filepaths(
                        self.data_dir,
                        match_regex=r"RC_\d{4}",
                        extension=".bz2",
                        recursive=True,
                    )
                )
            )
        else:
            return tuple()

    def download(
        self,
        *,
        date_range: Tuple[Optional[str], Optional[str]] = (None, None),
        force: bool = False,
    ) -> None:
        """
        Download 1 or more monthly Reddit comments files from archive.org
        and save them to disk under the ``data_dir`` directory.

        Args:
            date_range: Interval specifying the [start, end) dates for which
                comments files will be downloaded. Each item must be a str
                formatted as YYYY-MM or YYYY-MM-DD (the latter is converted
                to the corresponding YYYY-MM value). Both start and end values
                must be specified, but a null value for either is automatically
                replaced by the minimum or maximum valid values, respectively.
            force: If True, download the dataset, even if it already
                exists on disk under ``data_dir``.
        """
        date_range = utils.validate_and_clip_range(
            date_range, self.full_date_range, val_type=(str, bytes)
        )
        filestubs = self._generate_filestubs(date_range)
        for filestub in filestubs:
            tio.download_file(
                urllib.parse.urljoin(DOWNLOAD_ROOT, filestub),
                filename=filestub,
                dirpath=self.data_dir,
                force=force,
            )

    def _generate_filestubs(self, date_range):
        """
        Generate a list of monthly filepath stubs in the interval [start, end),
        each with format "YYYY/RC_YYYY-MM.bz2".
        """
        fstubs = []
        start = self._parse_date(date_range[0])
        end = self._parse_date(date_range[1])
        for tot_mo in range(self._total_mos(start) - 1, self._total_mos(end) - 1):
            yr, mo = divmod(tot_mo, 12)
            fstubs.append(datetime(yr, mo + 1, 1).strftime("%Y/RC_%Y-%m.bz2"))
        return tuple(fstubs)

    def _parse_date(self, dt):
        """dt (str) => datetime"""
        try:
            return datetime.strptime(dt, "%Y-%m")
        except ValueError:
            return datetime.strptime(dt, "%Y-%m-%d")

    def _total_mos(self, dt):
        """dt (datetime) => int"""
        return dt.month + 12 * dt.year

    def __iter__(self):
        # for performance reasons, only iterate over files that are requested
        if self._date_range is not None:
            filepaths = [
                self.data_dir.joinpath(filestub)
                for filestub in self._generate_filestubs(self._date_range)
            ]
            for filepath in filepaths:
                if not filepath.is_file():
                    raise OSError(
                        f"requested comments file {filepath} not found;\n"
                        "has the dataset been downloaded yet?"
                    )
        else:
            filepaths = self.filepaths
            if not filepaths:
                raise OSError(
                    f"no comments files found in {self.data_dir} directory;\n"
                    "has the dataset been downloaded yet?"
                )

        for filepath in filepaths:
            for line in tio.read_json(filepath, mode="rb", lines=True):
                line["created_utc"] = self._convert_timestamp(
                    line.get("created_utc", "")
                )
                line["retrieved_on"] = self._convert_timestamp(
                    line.get("retrieved_on", "")
                )
                line["body"] = self._clean_content(line["body"])
                yield line

    def _get_filters(self, subreddit, date_range, score_range, min_len):
        filters = []
        if min_len is not None:
            if min_len < 1:
                raise ValueError("`min_len` must be at least 1")
            filters.append(lambda record: len(record.get("body", "")) >= min_len)
        if subreddit is not None:
            subreddit = utils.validate_set_members(subreddit, (str, bytes))
            filters.append(lambda record: record.get("subreddit") in subreddit)
        if date_range is not None:
            date_range = utils.validate_and_clip_range(
                date_range, self.full_date_range, val_type=(str, bytes)
            )
            filters.append(
                lambda record: (
                    record.get("created_utc")
                    and date_range[0] <= record["created_utc"] < date_range[1]
                )
            )
        if score_range is not None:
            score_range = utils.validate_and_clip_range(
                score_range, self._full_score_range, val_type=(int, float)
            )
            filters.append(
                lambda record: (
                    record.get("score")
                    and score_range[0] <= record["score"] < score_range[1]
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
        subreddit: Optional[Union[str, Set[str]]] = None,
        date_range: Optional[Tuple[Optional[str], Optional[str]]] = None,
        score_range: Optional[Tuple[Optional[int], Optional[int]]] = None,
        min_len: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Iterable[str]:
        """
        Iterate over comments (text-only) in 1 or more files of this dataset,
        optionally filtering by a variety of metadata and/or text length,
        in chronological order.

        Args:
            subreddit: Filter comments for those which were posted
                in the specified subreddit(s).
            date_range: Filter comments for those which were posted
                within the interval [start, end). Each item must be a str in
                ISO-standard format, i.e. some amount of YYYY-MM-DDTHH:mm:ss.
                Both start and end values must be specified, but a null value
                for either is automatically replaced by the minimum or maximum
                valid values, respectively.
            score_range: Filter comments for those whose score
                (# upvotes minus # downvotes) is within the interval [low, high).
                Both start and end values must be specified, but a null value
                for either is automatically replaced by the minimum or maximum
                valid values, respectively.
            min_len: Filter comments for those whose body length in chars is
                at least this long.
            limit: Maximum number of comments passing all filters to yield.
                If None, all comments are iterated over.

        Yields:
            Text of the next comment in dataset passing all filters.

        Raises:
            ValueError: If any filtering options are invalid.
        """
        self._date_range = date_range  # used to limit files iterated
        try:
            filters = self._get_filters(subreddit, date_range, score_range, min_len)
            for record in itertools.islice(self._filtered_iter(filters), limit):
                yield record["body"]
        finally:
            self._date_range = None

    def records(
        self,
        *,
        subreddit: Optional[Union[str, Set[str]]] = None,
        date_range: Optional[Tuple[Optional[str], Optional[str]]] = None,
        score_range: Optional[Tuple[Optional[int], Optional[int]]] = None,
        min_len: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Iterable[types.Record]:
        """
        Iterate over comments (including text and metadata) in 1 or more files
        of this dataset, optionally filtering by a variety of metadata and/or
        text length, in chronological order.

        Args:
            subreddit: Filter comments for those which were posted
                in the specified subreddit(s).
            date_range: Filter comments for those which were posted
                within the interval [start, end). Each item must be a str in
                ISO-standard format, i.e. some amount of YYYY-MM-DDTHH:mm:ss.
                Both start and end values must be specified, but a null value
                for either is automatically replaced by the minimum or maximum
                valid values, respectively.
            score_range: Filter comments for those whose score
                (# upvotes minus # downvotes) is within the interval [low, high).
                Both start and end values must be specified, but a null value
                for either is automatically replaced by the minimum or maximum
                valid values, respectively.
            min_len: Filter comments for those whose body length in chars is
                at least this long.
            limit: Maximum number of comments passing all filters to yield.
                If None, all comments are iterated over.

        Yields:
            Text of the next comment in dataset passing all filters,
            and its corresponding metadata.

        Raises:
            ValueError: If any filtering options are invalid.
        """
        self._date_range = date_range  # used to limit files iterated
        try:
            filters = self._get_filters(subreddit, date_range, score_range, min_len)
            for record in itertools.islice(self._filtered_iter(filters), limit):
                yield types.Record(text=record.pop("body"), meta=record)
        finally:
            self._date_range = None

    def _convert_timestamp(self, timestamp):
        try:
            return datetime.utcfromtimestamp(int(timestamp)).strftime(
                "%Y-%m-%dT%H:%M:%S"
            )
        except (ValueError, TypeError):
            return ""

    def _clean_content(self, content):
        # strip out link markup, e.g. [foo](http://foo.com)
        content = RE_REDDIT_LINK.sub(r"\1", content)
        # clean up basic HTML cruft
        content = content.replace("&gt;", ">").replace("&lt;", "<")
        # strip out text markup, e.g. * for bold text
        content = content.replace("`", "").replace("*", "").replace("~", "")
        # normalize whitespace
        return constants.RE_NONBREAKING_SPACE.sub(
            " ", constants.RE_LINEBREAK.sub(r"\n", content)
        ).strip()
