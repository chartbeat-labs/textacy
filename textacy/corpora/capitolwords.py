# -*- coding: utf-8 -*-
"""
The CapitolWords Corpus
-----------------------

Download to and stream from disk a corpus of (almost all) speeches given by the
main protagonists of the 2016 U.S. Presidential election that had previously
served in the U.S. Congress — including Hillary Clinton, Bernie Sanders, Barack
Obama, Ted Cruz, and John Kasich — from January 1996 through June 2016.

The corpus contains over 11k documents comprised of nearly 7M tokens. Each
document contains 7 fields:

    * text: full(?) text of the speech
    * title: title of the speech, in all caps
    * date: date on which the speech was given, as an ISO-standard string
    * speaker_name: first and last name of the speaker
    * speaker_party: political party of the speaker ('R' for Republican, 'D' for
      Democrat, and 'I' for Independent)
    * congress: number of the Congress in which the speech was given; ranges
      continuously between 104 and 114
    * chamber: chamber of Congress in which the speech was given; almost all are
      either 'House' or 'Senate', with a small number of 'Extensions'

This corpus was derived from the data provided by the Sunlight Foundation's
`Capitol Words API <http://sunlightlabs.github.io/Capitol-Words/>`_.
"""
import io
import logging
import os

import requests

from textacy import __resources_dir__
from textacy.compat import PY2, string_types
from textacy.fileio import make_dirs, read_json_lines

if PY2:
    URL = 'https://s3.amazonaws.com/chartbeat-labs/capitol-words-py2.json.gz'
else:
    URL = 'https://s3.amazonaws.com/chartbeat-labs/capitol-words-py3.json.gz'
FILENAME = URL.rsplit('/', 1)[-1]

MIN_DATE = '1996-01-01'
MAX_DATE = '2016-06-30'

LOGGER = logging.getLogger(__name__)


class CapitolWords(object):
    """
    Download data and stream from disk a collection of Congressional speeches
    that includes the full text and key metadata for each::

        >>> cw = textacy.corpora.CapitolWords()
        >>> for text in cw.texts(limit=10):
        ...     print(text)
        >>> for record in cw.records(limit=10):
        ...     print(record['title'], record['date'])
        ...     print(record['text'])

    Filter speeches by metadata and text length::

        >>> for record in cw.records(speaker_name='Bernie Sanders', limit=1):
        ...     print(record['date'], record['text'])
        >>> for record in cw.records(speaker_party='D', congress={110, 111, 112},
        ...                          chamber='Senate', limit=10):
        ...     print(record['speaker_name'], record['title'])
        >>> for record in cw.records(speaker_name={'Barack Obama', 'Hillary Clinton'},
        ...                          date_range=('2002-01-01', '2002-12-31')):
        ...     print(record['speaker_name'], record['title'], record['date'])
        >>> for text in cw.texts(min_len=50000):
        ...     print(len(text))

    Stream speeches into a `Corpus`::

        >>> text_stream, metadata_stream = textacy.fileio.split_record_fields(
        ...     cw.records(limit=100), 'text')
        >>> tc = textacy.Corpus.from_texts('en', text_stream, metadata_stream)
        >>> print(tc)

    Args:
        data_dir (str): path on disk containing corpus data; if None, textacy's
            default `__resources_dir__` is used
        download_if_missing (bool): if True and corpus data file isn't found on
            disk, download the file and save it to disk under `data_dir`

    Raises:
        OSError: if corpus data file isn't found under `data_dir` and
            `download_if_missing` is False

    Attributes:
        speaker_names (Set[str]): full names of all speakers included in corpus,
            e.g. `'Bernie Sanders'`
        speaker_parties (Set[str]): all distinct political parties of speakers,
            e.g. `'R'`
        chambers (Set[str]): all distinct chambers in which speeches were given,
            e.g. `'House'`
        congresses (Set[int]): all distinct numbers of the congresses in which
            speeches were given, e.g. `114`
    """

    speaker_names = {
        'Barack Obama', 'Bernie Sanders', 'Hillary Clinton', 'Jim Webb', 'Joe Biden',
        'John Kasich', 'Joseph Biden', 'Lincoln Chafee', 'Lindsey Graham', 'Marco Rubio',
        'Mike Pence', 'Rand Paul', 'Rick Santorum', 'Ted Cruz'}
    speaker_parties = {'R', 'D', 'I'}
    chambers = {'Extensions', 'House', 'Senate'}
    congresses = {104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114}

    def __init__(self, data_dir=None, download_if_missing=True):
        if data_dir is None:
            data_dir = __resources_dir__
        self.filepath = os.path.join(data_dir, 'capitolwords', FILENAME)
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

    def _iterate(self, text_only, speaker_name=None, speaker_party=None,
                 chamber=None, congress=None, date_range=None, min_len=None,
                 limit=-1):
        """Note: Use `.texts()` or `.records()` to iterate over corpus data."""
        # prepare filters
        if speaker_name:
            if isinstance(speaker_name, string_types):
                speaker_name = {speaker_name}
            if not all(item in self.speaker_names for item in speaker_name):
                raise ValueError(
                    'all values in `speaker_name` must be valid; see `CapitolWords.speaker_names`')
        if speaker_party:
            if isinstance(speaker_party, string_types):
                speaker_party = {speaker_party}
            if not all(item in self.speaker_parties for item in speaker_party):
                raise ValueError(
                    'all values in `speaker_party` must be valid; see `CapitolWords.speaker_parties`')
        if chamber:
            if isinstance(chamber, string_types):
                chamber = {chamber}
            if not all(item in self.chambers for item in chamber):
                raise ValueError(
                    'all values in `chamber` must be valid; see `CapitolWords.chambers`')
        if congress:
            if isinstance(congress, int):
                congress = {congress}
            if not all(item in self.congresses for item in congress):
                raise ValueError(
                    'all values in `congress` must be valid; see `CapitolWords.congresses`')
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
            if speaker_name and line['speaker_name'] not in speaker_name:
                continue
            if speaker_party and line['speaker_party'] not in speaker_party:
                continue
            if chamber and line['chamber'] not in chamber:
                continue
            if congress and line['congress'] not in congress:
                continue
            if date_range and not date_range[0] <= line['date'] <= date_range[1]:
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

    def texts(self, speaker_name=None, speaker_party=None, chamber=None,
              congress=None, date_range=None, min_len=None, limit=-1):
        """
        Iterate over texts in the CapitolWords corpus, optionally filtering by
        a variety of metadata and/or text length, in order of date.

        Args:
            speaker_name (str or Set[str]): filter speeches by the speakers'
                name; see :meth:`speaker_names <CapitolWords.speaker_names>`
            speaker_party (str or Set[str]): filter speeches by the speakers'
                party; see :meth:`speaker_parties <CapitolWords.speaker_parties>`
            chamber (str or Set[str]): filter speeches by the chamber in which
                they were given; see :meth:`chambers <CapitolWords.chambers>`
            congress (int or Set[int]): filter speeches by the congress in which
                they were given; see :meth:`congresses <CapitolWords.congresses>`
            date_range (List[str] or Tuple[str]): filter speeches by the date on
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
            True, speaker_name=speaker_name, speaker_party=speaker_party,
            chamber=chamber, congress=congress, date_range=date_range,
            min_len=min_len, limit=limit)
        for text in texts:
            yield text

    def records(self, speaker_name=None, speaker_party=None, chamber=None,
                congress=None, date_range=None, min_len=None, limit=-1):
        """
        Iterate over documents (including text and metadata) in the CapitolWords
        corpus, optionally filtering by a variety of metadata and/or text length,
        in order of date.

        Args:
            speaker_name (str or Set[str]): filter speeches by the speakers'
                name; see :meth:`speaker_names <CapitolWords.speaker_names>`
            speaker_party (str or Set[str]): filter speeches by the speakers'
                party; see :meth:`speaker_parties <CapitolWords.speaker_parties>`
            chamber (str or Set[str]): filter speeches by the chamber in which
                they were given; see :meth:`chambers <CapitolWords.chambers>`
            congress (int or Set[int]): filter speeches by the congress in which
                they were given; see :meth:`congresses <CapitolWords.congresses>`
            date_range (List[str] or Tuple[str]): filter speeches by the date on
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
        records = self._iterate(
            False, speaker_name=speaker_name, speaker_party=speaker_party,
            chamber=chamber, congress=congress, date_range=date_range,
            min_len=min_len, limit=limit)
        for record in records:
            yield record
