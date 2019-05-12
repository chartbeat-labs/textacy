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
from .. import constants
from .. import io as tio
from . import utils
from .dataset import Dataset

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
    Stream a collection of Congressional speeches from a compressed json file on disk,
    either as texts or text + metadata pairs.

    Download the data (one time only!) from the textacy-data repo
    (https://github.com/bdewilde/textacy-data), and save its contents to disk::

        >>> cw = CapitolWords()
        >>> cw.download()
        >>> cw.info
        {'name': 'capitol_words',
         'site_url': 'http://sunlightlabs.github.io/Capitol-Words/',
         'description': 'Collection of ~11k speeches in the Congressional Record given by notable U.S. politicians between Jan 1996 and Jun 2016.'}

    Iterate over speeches as texts or records with both text and metadata::

        >>> for text in cw.texts(limit=3):
        ...     print(text, end="\\n\\n")
        >>> for text, meta in cw.records(limit=3):
        ...     print("\\n{} ({})\\n{}".format(meta["title"], meta["speaker_name"], text))

    Filter speeches by a variety of metadata fields and text length::

        >>> for text, meta in cw.records(speaker_name="Bernie Sanders", limit=3):
        ...     print("\\n{}, {}\\n{}".format(meta["title"], meta["date"], text))
        >>> for text, meta in cw.records(speaker_party="D", congress={110, 111, 112},
        ...                          chamber="Senate", limit=3):
        ...     print(meta["title"], meta["speaker_name"], meta["date"])
        >>> for text, meta in cw.records(speaker_name={"Barack Obama", "Hillary Clinton"},
        ...                              date_range=("2005-01-01", "2005-12-31")):
        ...     print(meta["title"], meta["speaker_name"], meta["date"])
        >>> for text in cw.texts(min_len=50000):
        ...     print(len(text))

    Stream speeches into a :class:`textacy.Corpus`::

        >>> textacy.Corpus("en", data=ota.records(limit=100))
        Corpus(100 docs; 70496 tokens)

    Args:
        data_dir (str): Path to directory on disk under which dataset data is stored,
            i.e. ``/path/to/data_dir/capitol_words`` .

    Attributes:
        full_date_range (Tuple[str]): First and last dates for which speeches
            are available, each as an ISO-formatted string (YYYY-MM-DD).
        speaker_names (Set[str]): Full names of all speakers included in corpus,
            e.g. "Bernie Sanders".
        speaker_parties (Set[str]): All distinct political parties of speakers,
            e.g. "R".
        chambers (Set[str]): All distinct chambers in which speeches were given,
            e.g. "House".
        congresses (Set[int]): All distinct numbers of the congresses in which
            speeches were given, e.g. 114.
    """

    full_date_range = ("1996-01-01", "2016-06-30")
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

    def __init__(self, data_dir=os.path.join(constants.DEFAULT_DATA_DIR, NAME)):
        super(CapitolWords, self).__init__(NAME, meta=META)
        self.data_dir = data_dir
        self._filename = "capitol-words-py{py_version}.json.gz".format(
            py_version=2 if compat.PY2 else 3)
        self._filepath = os.path.join(self.data_dir, self._filename)

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
        filepath = utils.download_file(
            url,
            filename=self._filename,
            dirpath=self.data_dir,
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
        if min_len is not None:
            if min_len < 1:
                raise ValueError("`min_len` must be at least 1")
            filters.append(
                lambda record: len(record.get("text", "")) >= min_len
            )
        if date_range is not None:
            date_range = utils.validate_and_clip_range_filter(
                date_range, self.full_date_range, val_type=compat.string_types)
            filters.append(
                lambda record: (
                    record.get("date")
                    and date_range[0] <= record["date"] < date_range[1]
                )
            )
        if speaker_name is not None:
            speaker_name = utils.validate_set_member_filter(
                speaker_name, compat.string_types, valid_vals=self.speaker_names)
            filters.append(lambda record: record.get("speaker_name") in speaker_name)
        if speaker_party is not None:
            speaker_party = utils.validate_set_member_filter(
                speaker_party, compat.string_types, valid_vals=self.speaker_parties)
            filters.append(lambda record: record.get("speaker_party") in speaker_party)
        if chamber is not None:
            chamber = utils.validate_set_member_filter(
                chamber, compat.string_types, valid_vals=self.chambers)
            filters.append(lambda record: record.get("chamber") in chamber)
        if congress is not None:
            congress = utils.validate_set_member_filter(
                congress, int, valid_vals=self.congresses)
            filters.append(lambda record: record.get("congress") in congress)
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
