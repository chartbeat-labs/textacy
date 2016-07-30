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
    URL = 'https://cdn.rawgit.com/chartbeat-labs/textacy/5156197facb98f767f5d92ffa299d266bd50cf6b/data/capitol-words-py2.json.gz'
else:
    URL = 'https://cdn.rawgit.com/chartbeat-labs/textacy/5156197facb98f767f5d92ffa299d266bd50cf6b/data/capitol-words-py3.json.gz'
FILENAME = URL.rsplit('/', 1)[-1]

LOGGER = logging.getLogger(__name__)


class CapitolWords(object):
    """
    TODO: usage example.

    Args:
        data_dir (str)

    Attributes:
        speaker_names (set[str]): full names of all speakers included in corpus,
            e.g. `'Bernie Sanders'`
        speaker_parties (set[str]): all distinct political parties of speakers,
            e.g. `'R'`
        chambers (set[str]): all distinct chambers in which speeches were given,
            e.g. `'House'`
        congresses (set[int]): all distinct numbers of the congresses in which
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
            _data_dir = __resources_dir__
        self.filepath = os.path.join(_data_dir, 'capitolwords', FILENAME)
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

    def __iter__(self):
        mode = 'rb' if PY2 else 'rt'
        for line in read_json_lines(self.filepath, mode=mode):
            yield line

    def texts(self, speaker_name=None, speaker_party=None, chamber=None,
              congress=None, date_range=None, min_len=None, limit=-1):
        """
        Iterate over texts in the CapitolWords corpus, optionally filtering by
        a variety of metadata and/or text length.

        Args:
            speaker_name (str or set[str])
            speaker_party (str or set[str])
            chamber (str or set[str])
            congress (int or set[int])
            date_range (list[str] or tuple[str])
            min_len (int)
            limit (int)

        Yields:
            str

        Raises:
            ValueError
        """
        # prepare filters
        if speaker_name:
            if isinstance(speaker_name, string_types):
                speaker_name = {speaker_name}
            if not all(item in self.speaker_names for item in speaker_name):
                raise ValueError()
        if speaker_party:
            if isinstance(speaker_party, string_types):
                speaker_party = {speaker_party}
            if not all(item in self.speaker_parties for item in speaker_party):
                raise ValueError()
        if chamber:
            if isinstance(chamber, string_types):
                chamber = {chamber}
            if not all(item in self.chambers for item in chamber):
                raise ValueError()
        if congress:
            if isinstance(congress, int):
                congress = {congress}
            if not all(item in self.congresses for item in congress):
                raise ValueError()
        if date_range:
            if not isinstance(date_range, (list, tuple)):
                raise ValueError()
            if not len(date_range) == 2:
                raise ValueError()

        n = 0
        for doc in self:

            if speaker_name and doc['speaker_name'] not in speaker_name:
                continue
            if speaker_party and doc['speaker_party'] not in speaker_party:
                continue
            if chamber and doc['chamber'] not in chamber:
                continue
            if congress and doc['congress'] not in congress:
                continue
            if date_range and not date_range[0] <= doc['date'] <= date_range[1]:
                continue
            if min_len and len(doc['text']) < min_len:
                continue

            yield doc['text']

            n += 1
            if n == limit:
                break

    def docs(self, speaker_name=None, speaker_party=None, chamber=None,
             congress=None, date_range=None, min_len=None, limit=-1):
        """
        Iterate over documents (including text and metadata) in the CapitolWords
        corpus, optionally filtering by a variety of metadata and/or text length.

        Args:
            speaker_name (str or set[str])
            speaker_party (str or set[str])
            chamber (str or set[str])
            congress (int or set[int])
            date_range (list[str] or tuple[str])
            min_len (int)
            limit (int)

        Yields:
            str

        Raises:
            ValueError
        """
        # prepare filters
        if speaker_name:
            if isinstance(speaker_name, string_types):
                speaker_name = {speaker_name}
            if not all(item in self.speaker_names for item in speaker_name):
                raise ValueError()
        if speaker_party:
            if isinstance(speaker_party, string_types):
                speaker_party = {speaker_party}
            if not all(item in self.speaker_parties for item in speaker_party):
                raise ValueError()
        if chamber:
            if isinstance(chamber, string_types):
                chamber = {chamber}
            if not all(item in self.chambers for item in chamber):
                raise ValueError()
        if congress:
            if isinstance(congress, int):
                congress = {congress}
            if not all(item in self.congresses for item in congress):
                raise ValueError()
        if date_range:
            if not isinstance(date_range, (list, tuple)):
                raise ValueError()
            if not len(date_range) == 2:
                raise ValueError()

        n = 0
        for doc in self:

            if speaker_name and doc['speaker_name'] not in speaker_name:
                continue
            if speaker_party and doc['speaker_party'] not in speaker_party:
                continue
            if chamber and doc['chamber'] not in chamber:
                continue
            if congress and doc['congress'] not in congress:
                continue
            if date_range and not date_range[0] <= doc['date'] <= date_range[1]:
                continue
            if min_len and len(doc['text']) < min_len:
                continue

            yield doc

            n += 1
            if n == limit:
                break
