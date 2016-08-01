# -*- coding: utf-8 -*-
"""
The Supreme Court Corpus
------------------------

Download to and stream from disk a corpus of all decisions issued by the U.S.
Supreme Court from 1792 through 2016. That amounts to 23k documents and 120.5M
tokens. Each document contains 5 fields:

    * text: full text of the Court's decision
    * title: title of the court case, in all caps
    * docket: docket number of the court case, which some may find useful
    * decided_date: date on which the decision was given, as an ISO-standard string
    * argued_date: date on which the case was argued before the Supreme Court,
        as an ISO-standard string (NOTE: about half are missing this value)

This dataset was derived from `FindLaw's searchable database <http://caselaw.findlaw.com/court/us-supreme-court>`_
of court cases. Its creation was inspired by `this blog post <http://www.emilyinamillion.me/blog/2016/7/13/visualizing-supreme-court-topics-over-time>`_
by Emily Barry.
"""
import io
import logging
import os

import requests

from textacy import __resources_dir__
from textacy.compat import PY2, string_types
from textacy.fileio import make_dirs, read_json_lines

if PY2:
    URL = 'https://s3.amazonaws.com/chartbeat-labs/supreme-court-cases-py2.json.gz'
else:
    URL = 'https://s3.amazonaws.com/chartbeat-labs/supreme-court-cases-py3.json.gz'
FILENAME = URL.rsplit('/', 1)[-1]

MIN_DATE = '1792-08-01'
MAX_DATE = '2016-07-18'

LOGGER = logging.getLogger(__name__)

# TODO: Consider joining data with http://supremecourtdatabase.org/index.php


class SupremeCourt(object):

    def __init__(self, data_dir=None, download_if_missing=True):
        if data_dir is None:
            data_dir = __resources_dir__
        self.filepath = os.path.join(data_dir, 'supremecourt', FILENAME)
        if not os.path.exists(self.filepath):
            if download_if_missing is True:
                self._download_data()
            else:
                raise OSError('file "{}" not found'.format(self.filepath))

    def _download_data(self):
        LOGGER.info('downloading data from "%s"', URL)
        response = requests.get(URL)
        make_dirs(self.filepath, 'wb')
        with io.open(self.filepath, mode='wb') as f:
            f.write(response.content)

    def _iterate(self, text_only, date_range=None, min_len=None, limit=-1):
        """Note: Use `.texts()` or `.docs()` to iterate over corpus data."""
        # prepare date range filter
        if date_range:
            if not isinstance(date_range, (list, tuple)):
                raise ValueError('`date_range` must be a list or tuple, not %s', type(date_range))
            if not len(date_range) == 2:
                raise ValueError('`date_range` must have both start and end values')
            if not date_range[0]:
                date_range = (MIN_DATE, date_range[1])
            if not date_range[1]:
                date_range = (date_range[0], MAX_DATE)

        n = 0
        mode = 'rb' if PY2 else 'rt'
        for line in read_json_lines(self.filepath, mode=mode):
            if date_range and not date_range[0] <= line['decided_date'] <= date_range[1]:
                continue
            if min_len and len(line['text']) < min_len:
                continue

            if text_only is True:
                yield line['text']
            else:
                yield line

            n += 1
            if n == limit:
                break

    def texts(self, date_range=None, min_len=None, limit=-1):
        """
        Iterate over texts in the CapitolWords corpus, optionally filtering by
        a variety of metadata and/or text length, in order of date.

        Args:
            date_range (list[str] or tuple[str]): filter speeches by the date on
                which they were given; both start and end date must be specified,
                but a null value for either will be replaced by the min/max date
                available in the corpus
            min_len (int): filter speeches by the length (number of characters)
                in their text content
            limit (int): return no more than `limit` speeches, in order of date

        Yields:
            str: full text of next (by chronological order) speech in corpus
                passing all filter params

        Raises:
            ValueError: if any filtering options are invalid
        """
        texts = self._iterate(
            True, date_range=date_range, min_len=min_len, limit=limit)
        for text in texts:
            yield text

    def docs(self, date_range=None, min_len=None, limit=-1):
        """
        Iterate over documents (including text and metadata) in the CapitolWords
        corpus, optionally filtering by a variety of metadata and/or text length,
        in order of date.

        Args:
            date_range (list[str] or tuple[str]): filter speeches by the date on
                which they were given; both start and end date must be specified,
                but a null value for either will be replaced by the min/max date
                available in the corpus
            min_len (int): filter speeches by the length (number of characters)
                in their text content
            limit (int): return no more than `limit` speeches, in order of date

        Yields:
            dict: full text and metadata of next (by chronological order) speech
                in corpus passing all filter params

        Raises:
            ValueError: if any filtering options are invalid
        """
        docs = self._iterate(
            False, date_range=date_range, min_len=min_len, limit=limit)
        for doc in docs:
            yield doc
