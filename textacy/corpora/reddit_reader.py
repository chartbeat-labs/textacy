"""
Reddit Corpus Reader
--------------------

Stream a corpus of up to ~1.6 billion Reddit comments posted from October 2007
until May 2015, as either plaintext strings or content + metadata dicts.

.. code-block:: pycon

    >>> rr = RedditReader('/path/to/RC_2015-01.bz2')
    >>> for text in rr.texts(limit=5):  # plaintext comments
    ...     print(text)
    >>> for comment in rr.comments(min_len=100, limit=1):  # parsed comments
    ...     print(comment.keys())
    ...     print(comment['body'])
    >>> rr = RedditReader(['/path/to/RC_2015-01.bz2', '/path/to/RC_2015-02.bz2'])

Raw data is downloadable from https://archive.org/details/2015_reddit_comments_corpus.
"""
import datetime
import logging
import json
import re

from textacy.compat import bzip_open, str
from textacy.preprocess import normalize_whitespace


LOGGER = logging.getLogger(__name__)
REDDIT_LINK_RE = re.compile(r'\[([^]]+)\]\(https?://[^\)]+\)')


class RedditReader(object):
    """
    Stream Reddit comments from standardized, compressed files on disk, either as
    plaintext strings or dict documents with both text content and metadata.
    Download the data from https://archive.org/details/2015_reddit_comments_corpus;
    the files are named as `RC_YYYY-MM.bz2`.

    Args:
        paths (sequence[str] or str): name or names of reddit comment file(s) from
            which to stream comments
    """

    def __init__(self, paths):
        if isinstance(paths, str):
            self.paths = (paths,)
        else:
            self.paths = tuple(paths)

    def __iter__(self):
        for path in self.paths:
            try:
                with bzip_open(path, mode='rt') as f:
                    for line in f:
                        yield json.loads(line)
            except ValueError:  # Python 2 sucks and can't open bzip in text mode
                with bzip_open(path, mode='rb') as f:
                    for line in f:
                        yield json.loads(line)

    def _clean_content(self, content):
        # strip out link markup, e.g. [foo](http://foo.com)
        content = REDDIT_LINK_RE.sub(r'\1', content)
        # clean up basic HTML cruft
        content = content.replace('&gt;', '>').replace('&lt;', '<')
        # strip out text markup, e.g. * for bold text
        content = content.replace('`', '').replace('*', '').replace('~', '')
        # normalize whitespace
        return normalize_whitespace(content)

    def _parse_comment(self, comment):
        # convert str/int timestamp fields into datetime objects
        for key in ('created_utc', 'retrieved_on'):
            try:
                comment[key] = datetime.datetime.utcfromtimestamp(int(comment[key]))
            except KeyError:
                pass
        # clean up comment's text content
        comment['body'] = self._clean_content(comment['body'])
        return comment

    def texts(self, min_len=0, limit=-1):
        """
        Iterate over the comments in 1 or more reddit comments files (RC_YYYY-MM.bz2),
        yielding the plain text of comments, one at a time.

        Args:
            min_len (int): minimum length in chars that a comment must have
                for it to be returned; too-short comments are skipped (optional)
            limit (int): maximum number of comments (passing `min_len`) to yield;
                if -1, all comments in the db file are iterated over (optional)

        Yields:
            str: plain text for the next comment in the reddit comments file
        """
        n_comments = 0
        for comment in self:
            text = self._clean_content(comment['body'])
            if len(text) < min_len:
                continue

            yield text

            n_comments += 1
            if n_comments == limit:
                break

    def comments(self, min_len=0, limit=-1):
        """
        Iterate over the comments in 1 or more reddit comments files (RC_YYYY-MM.bz2),
        yielding one (lightly parsed) comment at a time, as a dict.

        Args:
            min_len (int): minimum length in chars that a page must have
                for it to be returned; too-short pages are skipped
            limit (int): maximum number of pages (passing ``min_len``) to yield;
                if -1, all pages in the db dump are iterated over (optional)

        Yields:
            dict: the next comment in the reddit comments file; the comment's
                content is in the 'body' field, all other fields are metadata
        """
        n_comments = 0
        for comment in self:
            comment = self._parse_comment(comment)
            if len(comment['body']) < min_len:
                continue

            yield comment

            n_comments += 1
            if n_comments == limit:
                break
