# -*- coding: utf-8 -*-
"""
Capitol Words
-------------

A collection of ~11k (almost all) speeches given by the main protagonists of the
2016 U.S. Presidential election that had previously served in the U.S. Congress --
including Hillary Clinton, Bernie Sanders, Barack Obama, Ted Cruz, and John Kasich --
from January 1996 through June 2016.

Records include the following data:

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
from __future__ import absolute_import, division, print_function, unicode_literals

import itertools
import logging
import os

from .. import compat
from .. import data_dir as DATA_DIR
from .. import io as tio
from .dataset import Dataset, _download, _parse_date_range

LOGGER = logging.getLogger(__name__)

NAME = "capitol_words"
META = {
    "site_url": "http://sunlightlabs.github.io/Capitol-Words/",
    "description": (
        "Collection of ~11k speeches in the Congressional Record given by "
        "notable U.S. politicians between Jan 1996 and Jun 2016."
    ),
}
DOWNLOAD_ROOT = "https://github.com/bdewilde/textacy-data/releases/download/"


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
        super(CapitolWords, self).__init__(NAME, meta=META)
        self._data_dir = os.path.join(data_dir, NAME)
        self._filename = "capitol-words-py{py_version}.json.gz".format(
            py_version=2 if compat.PY2 else 3)
        self._filepath = os.path.join(self._data_dir, self._filename)

    @property
    def filepath(self):
        """
        str: Full path on disk for CapitolWords data as compressed json file.
            ``None`` if file is not found, e.g. has not yet been downloaded.
        """
        if os.path.isfile(self._filepath):
            return self._filepath
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
            py_version=2 if compat.PY2 else 3,
            data_version=1.0,
        )
        url = compat.urljoin(DOWNLOAD_ROOT, release_tag + "/" + self._filename)
        filepath = _download(
            url,
            filename=self._filename,
            dirpath=self._data_dir,
            force=force,
        )

    def __iter__(self):
        if not os.path.isfile(self._filepath):
            raise OSError(
                "dataset file {} not found;\n"
                "has the dataset been downloaded yet?".format(self._filepath)
            )
        mode = "rb" if compat.PY2 else "rt"  # TODO: check this
        for record in tio.read_json(self._filepath, mode=mode, lines=True):
            yield record

    def _get_filters(
        self,
        speaker_name,
        speaker_party,
        chamber,
        congress,
        date_range,
        min_len,
    ):
        filters = []

        def get_filter(filter_vals, vals_type, record_field, class_attr):
            if filter_vals is not None:
                if isinstance(filter_vals, vals_type):
                    filter_vals = {filter_vals}
                if not all(val in getattr(self, class_attr) for val in filter_vals):
                    raise ValueError(
                        "not all values in `{record_field}` are valid; "
                        "see :attr:`CapitolWords.{class_attr}`".format(
                            record_field=record_field,
                            class_attr=class_attr,
                        )
                    )
                return lambda record: record.get(record_field) in filter_vals
            else:
                return None

        if min_len is not None:
            if min_len < 1:
                raise ValueError("`min_len` must be at least 1")
            filters.append(
                lambda record: len(record.get("text", "")) >= min_len
            )
        if date_range is not None:
            date_range = _parse_date_range(date_range, self.min_date, self.max_date)
            filters.append(
                lambda record: record.get("year") and date_range[0] <= record["year"] < date_range[1]
            )
        candidate_filters = [
            (speaker_name, compat.string_types, "speaker_name", "speaker_names"),
            (speaker_party, compat.string_types, "speaker_party", "speaker_parties"),
            (chamber, compat.string_types, "chamber", "chambers"),
            (congress, int, "congress", "congresses"),
        ]
        for candidate_filter in candidate_filters:
            filter_ = get_filter(*candidate_filter)
            if filter_:
                filters.append(filter_)
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
        speaker_name=None,
        speaker_party=None,
        chamber=None,
        congress=None,
        date_range=None,
        min_len=None,
        limit=None,
    ):
        """
        Iterate over speeches in this dataset, optionally filtering by a variety
        of metadata and/or text length, and yield texts only,
        in chronological order.

        Args:
            speaker_name (str or Set[str]): Filter speeches by the speakers' name;
                see :attr:`CapitolWords.speaker_names`.
            speaker_party (str or Set[str]): Filter speeches by the speakers'
                party; see :attr:`CapitolWords.speaker_parties`.
            chamber (str or Set[str]): Filter speeches by the chamber in which
                they were given; see :attr:`CapitolWords.chambers`.
            congress (int or Set[int]): Filter speeches by the congress in which
                they were given; see :attr:`CapitolWords.congresses`.
            date_range (List[str] or Tuple[str]): Filter speeches by the date on
                which they were given. Both start and end date must be specified,
                but a null value for either will be replaced by the min/max date
                available for the dataset.
            min_len (int): Filter speeches by the length (number of characters)
                of their text content.
            limit (int): Yield no more than ``limit`` speeches that match all
                specified filters.

        Yields:
            str: Full text of next (by chronological order) speech in dataset
            passing all filter params.

        Raises:
            ValueError: If any filtering options are invalid.
        """
        filters = self._get_filters(
            speaker_name, speaker_party, chamber, congress, date_range, min_len)
        for record in itertools.islice(self._filtered_iter(filters), limit):
            yield record["text"]

    def records(
        self,
        speaker_name=None,
        speaker_party=None,
        chamber=None,
        congress=None,
        date_range=None,
        min_len=None,
        limit=None,
    ):
        """
        Iterate over speeches in this dataset, optionally filtering by a variety
        of metadata and/or text length, and yield text + metadata pairs,
        in chronological order.

        Args:
            speaker_name (str or Set[str]): Filter speeches by the speakers' name;
                see :attr:`CapitolWords.speaker_names`.
            speaker_party (str or Set[str]): Filter speeches by the speakers'
                party; see :attr:`CapitolWords.speaker_parties`.
            chamber (str or Set[str]): Filter speeches by the chamber in which
                they were given; see :attr:`CapitolWords.chambers`.
            congress (int or Set[int]): Filter speeches by the congress in which
                they were given; see :attr:`CapitolWords.congresses`.
            date_range (List[str] or Tuple[str]): Filter speeches by the date on
                which they were given. Both start and end date must be specified,
                but a null value for either will be replaced by the min/max date
                available for the dataset.
            min_len (int): Filter speeches by the length (number of characters)
                of their text content.
            limit (int): Yield no more than ``limit`` speeches that match all
                specified filters.

        Yields:
            str: Text of the next speech in dataset passing all filters.
            dict: Metadata of the next speech in dataset passing all filters.

        Raises:
            ValueError: If any filtering options are invalid.
        """
        filters = self._get_filters(
            speaker_name, speaker_party, chamber, congress, date_range, min_len)
        for record in itertools.islice(self._filtered_iter(filters), limit):
            yield record.pop("text"), record
