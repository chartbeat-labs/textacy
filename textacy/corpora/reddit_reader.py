"""
"""
import bz2
import re

import ujson

from textacy.preprocess import normalize_whitespace


REDDIT_LINK_RE = re.compile(r'\[([^]]+)\]\(https?://[^\)]+\)')


class RedditReader(object):

    def __init__(self, path):
        self.path = path

    def __iter__(self):
        with bz2.BZ2File(self.path) as f:
            for line in f:
                yield ujson.loads(line)

    def _clean_content(self, content):
        # strip out link markup, e.g. [foo](http://foo.com)
        content = REDDIT_LINK_RE.sub(r'\1', content)
        # clean up basic HTML cruft
        content = content.replace('&gt;', '>').replace('&lt;', '<')
        # strip out text markup, e.g. * for bold text
        content = content.replace('`', '').replace('*', '').replace('~', '')
        # normalize whitespace
        return normalize_whitespace(content)

    def texts(self, min_len=0, limit=-1):
        """
        Iterate over the comments in a reddit comments file (RC_*.bz2),
        yielding the plain text of a comment, one at a time.

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
        Iterate over the comments in a reddit comments file (RC_*.bz2),
        yielding one comment at a time, as a dict.

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
            comment['body'] = self._clean_content(comment['body'])
            if len(comment['body']) < min_len:
                continue

            yield comment

            n_comments += 1
            if n_comments == limit:
                break
