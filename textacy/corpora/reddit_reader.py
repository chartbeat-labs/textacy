"""
Reddit Corpus Reader
--------------------

Stream a corpus of up to ~1.6 billion Reddit comments posted from October 2007
until May 2015, as either texts (str) or records (dict) with both content and
metadata.

Key fields in each record are as follows:

    * ``body``: full text of the comment
    * ``created_utc``: date on which the comment was posted
    * ``subreddit``: sub-reddit in which the comment was posted, excluding the
      familiar '/r/' prefix
    * ``score``: net score (upvotes - downvotes) on the comment
    * ``gilded``: number of times this comment received reddit gold

Raw data is downloadable from https://archive.org/details/2015_reddit_comments_corpus.
"""
from datetime import datetime
import logging
import os
import re

from textacy.compat import PY2, string_types
from textacy.fileio import read_json_lines
from textacy.preprocess import normalize_whitespace


LOGGER = logging.getLogger(__name__)
REDDIT_LINK_RE = re.compile(r'\[([^]]+)\]\(https?://[^\)]+\)')
MIN_DATE = '0001-01-01'
MAX_DATE = '9999-12-31'
MIN_INT = -2147483647
MAX_INT = 2147483647


class RedditReader(object):
    """
    Stream Reddit comments from standardized, compressed files on disk, either
    as texts (str) or records (dict) with both text content and metadata.
    Download the data from https://archive.org/details/2015_reddit_comments_corpus;
    the files are named as `RC_YYYY-MM.bz2`.

    .. code-block:: pycon

        >>> rr = RedditReader('RC_2015-01.bz2')
        >>> for text in rr.texts(limit=5):  # plaintext comments
        ...     print(text)
        >>> for record in rr.records(min_len=100, limit=1):  # parsed comments
        ...     print(record['subreddit'], record['created_utc'])
        ...     print(record['body'])
        >>> for record in rr.records(subreddit='leagueoflegends', limit=10):
        ...     print(record['score'], record['body'])
        >>> for record in rr.records(date_range=('2015-01-01T00:00:00', '2015-01-01T23:59:59'),
        ...                          limit=100):
        ...     print(record['created_utc'])
        >>> rr = RedditReader(['RC_2015-01.bz2', 'RC_2015-02.bz2'])

    Args:
        paths (str or Sequence[str]): name(s) of reddit comment file(s) from
            which to stream comments
    """
    def __init__(self, paths):
        if isinstance(paths, string_types):
            self.paths = (paths,)
        elif isinstance(paths, (list, tuple, set, frozenset)):
            self.paths = tuple(sorted(paths))
        else:
            raise ValueError('`paths` type "{}" not valid'.format(type(paths)))

    def __repr__(self):
        first_path = os.path.split(self.paths[0])[-1]
        s = 'RedditReader("{}"{})'.format(
            first_path, ', ...' if len(self.paths) > 1 else '')
        return s

    def _clean_content(self, content):
        # strip out link markup, e.g. [foo](http://foo.com)
        content = REDDIT_LINK_RE.sub(r'\1', content)
        # clean up basic HTML cruft
        content = content.replace('&gt;', '>').replace('&lt;', '<')
        # strip out text markup, e.g. * for bold text
        content = content.replace('`', '').replace('*', '').replace('~', '')
        # normalize whitespace
        return normalize_whitespace(content)

    def _convert_timestamp(self, timestamp):
        try:
            return datetime.utcfromtimestamp(int(timestamp)).strftime('%Y-%m-%dT%H:%M:%S')
        except (ValueError, TypeError):
            return ''

    def _iterate(self, text_only, subreddit, date_range, score_range,
                 min_len, limit):
        """Note: Use `.texts()` or `.records()` to iterate over corpus data."""
        if subreddit:
            if isinstance(subreddit, string_types):
                subreddit = {subreddit}
            elif isinstance(subreddit, (list, tuple)):
                subreddit = set(subreddit)
        if date_range:
            if not isinstance(date_range, (list, tuple)):
                msg = '`date_range` must be a list or tuple, not {}'.format(
                    type(date_range))
                raise ValueError(msg)
            if not len(date_range) == 2:
                msg = '`date_range` must have both start and end values'
                raise ValueError(msg)
            if not date_range[0]:
                date_range = (MIN_DATE, date_range[1])
            if not date_range[1]:
                date_range = (date_range[0], MAX_DATE)
        if score_range:
            if not isinstance(score_range, (list, tuple)):
                msg = '`score_range` must be a list or tuple, not {}'.format(
                    type(score_range))
                raise ValueError(msg)
            if len(score_range) != 2:
                msg = '`score_range` must have both min and max values'
                raise ValueError(msg)
            if not score_range[0]:
                score_range = (MIN_INT, score_range[1])
            if not score_range[1]:
                score_range = (score_range[0], MAX_INT)

        n = 0
        mode = 'rb' if PY2 else 'rt'  # Python 2 can't open json in text mode
        for path in self.paths:
            for line in read_json_lines(path, mode=mode):

                if subreddit and line['subreddit'] not in subreddit:
                    continue
                if score_range and not score_range[0] <= line['score'] <= score_range[1]:
                    continue
                line['created_utc'] = self._convert_timestamp(line.get('created_utc', ''))
                if date_range and not date_range[0] <= line['created_utc'] <= date_range[1]:
                    continue
                line['body'] = self._clean_content(line['body'])
                if min_len and len(line['body']) < min_len:
                    continue

                if text_only is True:
                    yield line['body']
                else:
                    line['retrieved_on'] = self._convert_timestamp(line.get('retrieved_on', ''))
                    yield line

                n += 1
                if n == limit:
                    break

            if n == limit:
                break

    def texts(self, subreddit=None, date_range=None, score_range=None,
              min_len=0, limit=-1):
        """
        Iterate over the comments in 1 or more Reddit comments files
        (``RC_YYYY-MM.bz2``), yielding the plain text of comments, one at a time.

        Args:
            subreddit (str or Set[str]): filter comments by the subreddit in
                which they were posted
            date_range (List[str] or Tuple[str]): filter comments by the date on
                which they were posted; both start and end date must be specified,
                but a null value for either will effectively unbound the range
                on the corresponding side
            score_range (List[int] or Tuple[int]): filter comments by score
                (# upvotes minus # downvotes); both min and max score must be
                specified, but a null value for either will effectively unbound
                the range on the corresponding side
            min_len (int): minimum length in chars that a comment must have
                for it to be returned; too-short comments are skipped (optional)
            limit (int): maximum number of comments (passing `min_len`) to yield;
                if -1, all comments in the db file are iterated over (optional)

        Yields:
            str: plain text for the next comment in the reddit comments file
        """
        texts = self._iterate(
            True, subreddit=subreddit, date_range=date_range,
            score_range=score_range, min_len=min_len, limit=limit)
        for text in texts:
            yield text

    def records(self, subreddit=None, date_range=None, score_range=None,
                min_len=0, limit=-1):
        """
        Iterate over the comments in 1 or more Reddit comments files
        (``RC_YYYY-MM.bz2``), yielding one (lightly parsed) comment at a time,
        as a dict.

        Args:
            subreddit (str or Set[str]): filter comments by the subreddit in
                which they were posted
            date_range (List[str] or Tuple[str]): filter comments by the date on
                which they were posted; both start and end date must be specified,
                but a null value for either will effectively unbound the range
                on the corresponding side
            score_range (List[int] or Tuple[int]): filter comments by score
                (# upvotes minus # downvotes); both min and max score must be
                specified, but a null value for either will effectively unbound
                the range on the corresponding side
            min_len (int): minimum length in chars that a comment must have
                for it to be returned; too-short comments are skipped (optional)
            limit (int): maximum number of comments (passing `min_len`) to yield;
                if -1, all comments in the db file are iterated over (optional)

        Yields:
            str: plain text for the next comment in the reddit comments file
        """
        records = self._iterate(
            False, subreddit=subreddit, date_range=date_range,
            score_range=score_range, min_len=min_len, limit=limit)
        for record in records:
            yield record
