# -*- coding: utf-8 -*-
"""
Capitol Words
-------------

A collection of ~11k (almost all) speeches given by the main protagonists of the
2016 U.S. Presidential election that had previously served in the U.S. Congress --
including Hillary Clinton, Bernie Sanders, Barack Obama, Ted Cruz, and John Kasich --
from January 1996 through June 2016.

Records include the following fields:

    * ``text``: Full text of the Congressperson's remarks.
    * ``title``: Title of the speech, in all caps.
    * ``date``: Date on which the speech was given, as an ISO-standard string.
    * ``speaker_name``: First and last name of the speaker.
    * ``speaker_party``: Political party of the speaker: "R" for Republican,
      "D" for Democrat, "I" for Independent.
    * ``congress``: Number of the Congress in which the speech was given: ranges
      continuously between 104 and 114.
    * ``chamber``: Chamber of Congress in which the speech was given: almost all
      are either "House" or "Senate", with a small number of "Extensions".

This dataset was derived from data provided by the (now defunct) Sunlight
Foundation's `Capitol Words API <http://sunlightlabs.github.io/Capitol-Words/>`_.
"""
from __future__ import unicode_literals

import logging
import os

import requests

from .. import compat
from .. import data_dir
from .. import io
from .base import Dataset

LOGGER = logging.getLogger(__name__)

NAME = "capitol_words"
DESCRIPTION = (
    "Collection of ~11k speeches in the Congressional Record given by "
    "notable U.S. politicians between Jan 1996 and Jun 2016."
)
SITE_URL = "http://sunlightlabs.github.io/Capitol-Words/"  # TODO: change to propublica?
DOWNLOAD_ROOT = "https://github.com/bdewilde/textacy-data/releases/download/"
DATA_DIR = os.path.join(data_dir, NAME)


class CapitolWords(Dataset):
    """
    Stream Congressional speeches from a compressed json file on disk, either
    as texts (str) or records (dict) with both text content and metadata.

    Download a Python version-specific compressed json file from the
    `textacy-data <https://github.com/bdewilde/textacy-data>`_ repo::

        >>> cw = CapitolWords()
        >>> cw.download()
        >>> cw.info
        {'data_dir': 'path/to/textacy/data/capitolwords',
         'description': 'Collection of ~11k speeches in the Congressional Record given by notable U.S. politicians between Jan 1996 and Jun 2016.',
         'name': 'capitol_words',
         'site_url': 'http://sunlightlabs.github.io/Capitol-Words/'}

    Iterate over speeches as plain texts or records with both text and metadata::

        >>> for text in cw.texts(limit=5):
        ...     print(text)
        >>> for record in cw.records(limit=5):
        ...     print(record['title'], record['date'])
        ...     print(record['text'])

    Filter speeches by a variety of metadata fields and text length::

        >>> for record in cw.records(speaker_name='Bernie Sanders', limit=1):
        ...     print(record['date'], record['text'])
        >>> for record in cw.records(speaker_party='D', congress={110, 111, 112},
        ...                          chamber='Senate', limit=5):
        ...     print(record['speaker_name'], record['title'])
        >>> for record in cw.records(speaker_name={'Barack Obama', 'Hillary Clinton'},
        ...                          date_range=('2002-01-01', '2002-12-31')):
        ...     print(record['speaker_name'], record['title'], record['date'])
        >>> for text in cw.texts(min_len=50000):
        ...     print(len(text))

    Stream speeches into a :class:`textacy.Corpus`::

        >>> text_stream, metadata_stream = textacy.io.split_records(
        ...     cw.records(limit=100), 'text')
        >>> c = textacy.Corpus('en', texts=text_stream, metadatas=metadata_stream)
        >>> c
        Corpus(100 docs; 70500 tokens)

    Args:
        data_dir (str): Path to directory on disk under which the data
            (a compressed json file like ``capitol-words-py3.json.gz``) is stored.

    Attributes:
        min_date (str): Earliest date for which speeches are available, as an
            ISO-formatted string (YYYY-MM-DD).
        max_date (str): Latest date for which speeches are available, as an
            ISO-formatted string (YYYY-MM-DD).
        speaker_names (Set[str]): Full names of all speakers included in corpus,
            e.g. "Bernie Sanders".
        speaker_parties (Set[str]): All distinct political parties of speakers,
            e.g. "R".
        chambers (Set[str]): All distinct chambers in which speeches were given,
            e.g. "House".
        congresses (Set[int]): All distinct numbers of the congresses in which
            speeches were given, e.g. 114.
    """

    min_date = "1996-01-01"
    max_date = "2016-06-30"
    speaker_names = {
        "Barack Obama",
        "Bernie Sanders",
        "Hillary Clinton",
        "Jim Webb",
        "Joe Biden",
        "John Kasich",
        "Joseph Biden",
        "Lincoln Chafee",
        "Lindsey Graham",
        "Marco Rubio",
        "Mike Pence",
        "Rand Paul",
        "Rick Santorum",
        "Ted Cruz",
    }
    speaker_parties = {"D", "I", "R"}
    chambers = {"Extensions", "House", "Senate"}
    congresses = {104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114}

    def __init__(self, data_dir=DATA_DIR):
        super(CapitolWords, self).__init__(
            name=NAME, description=DESCRIPTION, site_url=SITE_URL, data_dir=data_dir
        )
        self.filestub = "capitol-words-py{py_version}.json.gz".format(
            py_version=2 if compat.is_python2 else 3
        )
        self._filename = os.path.join(data_dir, self.filestub)

    @property
    def filename(self):
        """
        str: Full path on disk for CapitolWords data as compressed json file.
            ``None`` if file is not found, e.g. has not yet been downloaded.
        """
        if os.path.isfile(self._filename):
            return self._filename
        else:
            return None

    def download(self, force=False):
        """
        Download the data as a Python version-specific compressed json file and
        save it to disk under the ``data_dir`` directory.

        Args:
            force (bool): If True, download the dataset, even if it already
                exists on disk under ``data_dir``.
        """
        release_tag = "capitol_words_py{py_version}_v{data_version}".format(
            py_version=2 if compat.is_python2 else 3, data_version=1.0
        )
        url = compat.urljoin(DOWNLOAD_ROOT, release_tag + "/" + self.filestub)
        fname = self._filename
        if os.path.isfile(fname) and force is False:
            LOGGER.warning("File %s already exists; skipping download...", fname)
            return
        LOGGER.info("Downloading data from %s and writing it to %s", url, fname)
        io.write_http_stream(
            url, fname, mode="wb", encoding=None, make_dirs=True, chunk_size=1024
        )

    def texts(
        self,
        speaker_name=None,
        speaker_party=None,
        chamber=None,
        congress=None,
        date_range=None,
        min_len=None,
        limit=-1,
    ):
        """
        Iterate over speeches (text-only) in this dataset, optionally filtering
        by a variety of metadata and/or text length, in chronological order.

        Args:
            speaker_name (str or Set[str]): Filter speeches by the speakers' name;
                see :attr:`speaker_names <CapitolWords.speaker_names>`.
            speaker_party (str or Set[str]): Filter speeches by the speakers' party;
                see :attr:`speaker_parties <CapitolWords.speaker_parties>`.
            chamber (str or Set[str]): Filter speeches by the chamber in which they
                were given; see :attr:`chambers <CapitolWords.chambers>`.
            congress (int or Set[int]): Filter speeches by the congress in which
                they were given; see :attr:`congresses <CapitolWords.congresses>`.
            date_range (List[str] or Tuple[str]): Filter speeches by the date on
                which they were given. Both start and end date must be specified,
                but a null value for either will be replaced by the min/max date
                available for the dataset.
            min_len (int): Filter speeches by the length (number of characters)
                of their text content.
            limit (int): Yield no more than ``limit`` speeches that match all
                specified filters, in chronological order.

        Yields:
            str: Full text of next (by chronological order) speech in dataset
            passing all filter params.

        Raises:
            ValueError: If any filtering options are invalid.
        """
        texts = self._iterate(
            True,
            speaker_name=speaker_name,
            speaker_party=speaker_party,
            chamber=chamber,
            congress=congress,
            date_range=date_range,
            min_len=min_len,
            limit=limit,
        )
        for text in texts:
            yield text

    def records(
        self,
        speaker_name=None,
        speaker_party=None,
        chamber=None,
        congress=None,
        date_range=None,
        min_len=None,
        limit=-1,
    ):
        """
        Iterate over speeches (text and metadata) in this dataset, optionally
        filtering by a variety of metadata and/or text length, in chronological order.

        Args:
            speaker_name (str or Set[str]): Filter speeches by the speakers' name;
                see :attr:`speaker_names <CapitolWords.speaker_names>`.
            speaker_party (str or Set[str]): Filter speeches by the speakers'
                party; see :attr:`speaker_parties <CapitolWords.speaker_parties>`.
            chamber (str or Set[str]): Filter speeches by the chamber in which
                they were given; see :attr:`chambers <CapitolWords.chambers>`.
            congress (int or Set[int]): Filter speeches by the congress in which
                they were given; see :attr:`congresses <CapitolWords.congresses>`.
            date_range (List[str] or Tuple[str]): Filter speeches by the date on
                which they were given. Both start and end date must be specified,
                but a null value for either will be replaced by the min/max date
                available for the dataset.
            min_len (int): Filter speeches by the length (number of characters)
                of their text content.
            limit (int): Yield no more than ``limit`` speeches that match all
                specified filters, in chronological order.

        Yields:
            dict: Full text and metadata of next (by chronological order) speech
            in dataset passing all filter params.

        Raises:
            ValueError: If any filtering options are invalid.
        """
        records = self._iterate(
            False,
            speaker_name,
            speaker_party,
            chamber,
            congress,
            date_range,
            min_len,
            limit,
        )
        for record in records:
            yield record

    def _iterate(
        self,
        text_only,
        speaker_name,
        speaker_party,
        chamber,
        congress,
        date_range,
        min_len,
        limit,
    ):
        """
        Low-level method to iterate over the records in this dataset. Used by
        :meth:`CapitolWords.texts()` and :meth:`CapitolWords.records()`.
        """
        if not self.filename:
            raise IOError("{} file not found".format(self._filename))

        if speaker_name:
            if isinstance(speaker_name, compat.string_types):
                speaker_name = {speaker_name}
            if not all(item in self.speaker_names for item in speaker_name):
                raise ValueError(
                    "all values in `speaker_name` must be valid; "
                    "see :attr:`CapitolWords.speaker_names`"
                )
        if speaker_party:
            if isinstance(speaker_party, compat.string_types):
                speaker_party = {speaker_party}
            if not all(item in self.speaker_parties for item in speaker_party):
                raise ValueError(
                    "all values in `speaker_party` must be valid; "
                    "see :attr:`CapitolWords.speaker_parties`"
                )
        if chamber:
            if isinstance(chamber, compat.string_types):
                chamber = {chamber}
            if not all(item in self.chambers for item in chamber):
                raise ValueError(
                    "all values in `chamber` must be valid; "
                    "see :attr:`CapitolWords.chambers`"
                )
        if congress:
            if isinstance(congress, int):
                congress = {congress}
            if not all(item in self.congresses for item in congress):
                raise ValueError(
                    "all values in `congress` must be valid; "
                    "see :attr:`CapitolWords.congresses`"
                )
        if date_range:
            date_range = self._parse_date_range(date_range)

        n = 0
        mode = "rb" if compat.is_python2 else "rt"  # TODO: check this
        for line in io.read_json(self.filename, mode=mode, lines=True):

            if speaker_name and line["speaker_name"] not in speaker_name:
                continue
            if speaker_party and line["speaker_party"] not in speaker_party:
                continue
            if chamber and line["chamber"] not in chamber:
                continue
            if congress and line["congress"] not in congress:
                continue
            if date_range and not date_range[0] <= line["date"] <= date_range[1]:
                continue
            if min_len and len(line["text"]) < min_len:
                continue

            if text_only is True:
                yield line["text"]
            else:
                yield line

            n += 1
            if n == limit:
                break
