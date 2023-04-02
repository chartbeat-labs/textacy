"""
Capitol Words Congressional speeches
------------------------------------

A collection of ~11k (almost all) speeches given by the main protagonists of the
2016 U.S. Presidential election that had previously served in the U.S. Congress --
including Hillary Clinton, Bernie Sanders, Barack Obama, Ted Cruz, and John Kasich --
from January 1996 through June 2016.

Records include the following data:

    - ``text``: Full text of the Congressperson's remarks.
    - ``title``: Title of the speech, in all caps.
    - ``date``: Date on which the speech was given, as an ISO-standard string.
    - ``speaker_name``: First and last name of the speaker.
    - ``speaker_party``: Political party of the speaker: "R" for Republican,
      "D" for Democrat, "I" for Independent.
    - ``congress``: Number of the Congress in which the speech was given: ranges
      continuously between 104 and 114.
    - ``chamber``: Chamber of Congress in which the speech was given: almost all
      are either "House" or "Senate", with a small number of "Extensions".

This dataset was derived from data provided by the (now defunct) Sunlight
Foundation's `Capitol Words API <http://sunlightlabs.github.io/Capitol-Words/>`_.
"""
from __future__ import annotations

import itertools
import logging
import urllib.parse
from typing import Any, Callable, ClassVar, Iterable, Optional

from .. import constants
from .. import io as tio
from .. import types, utils
from .base import Dataset


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

        >>> import textacy.datasets
        >>> ds = textacy.datasets.CapitolWords()
        >>> ds.download()
        >>> ds.info
        {'name': 'capitol_words',
         'site_url': 'http://sunlightlabs.github.io/Capitol-Words/',
         'description': 'Collection of ~11k speeches in the Congressional Record given by notable U.S. politicians between Jan 1996 and Jun 2016.'}

    Iterate over speeches as texts or records with both text and metadata::

        >>> for text in ds.texts(limit=3):
        ...     print(text, end="\\n\\n")
        >>> for text, meta in ds.records(limit=3):
        ...     print("\\n{} ({})\\n{}".format(meta["title"], meta["speaker_name"], text))

    Filter speeches by a variety of metadata fields and text length::

        >>> for text, meta in ds.records(speaker_name="Bernie Sanders", limit=3):
        ...     print("\\n{}, {}\\n{}".format(meta["title"], meta["date"], text))
        >>> for text, meta in ds.records(speaker_party="D", congress={110, 111, 112},
        ...                          chamber="Senate", limit=3):
        ...     print(meta["title"], meta["speaker_name"], meta["date"])
        >>> for text, meta in ds.records(speaker_name={"Barack Obama", "Hillary Clinton"},
        ...                              date_range=("2005-01-01", "2005-12-31")):
        ...     print(meta["title"], meta["speaker_name"], meta["date"])
        >>> for text in ds.texts(min_len=50000):
        ...     print(len(text))

    Stream speeches into a :class:`textacy.Corpus <textacy.corpus.Corpus>`::

        >>> textacy.Corpus("en", data=ota.records(limit=100))
        Corpus(100 docs; 70496 tokens)

    Args:
        data_dir: Path to directory on disk under which dataset is stored,
            i.e. ``/path/to/data_dir/capitol_words``.

    Attributes:
        full_date_range: First and last dates for which speeches are available,
            each as an ISO-formatted string (YYYY-MM-DD).
        speaker_names: Full names of all speakers included in corpus, e.g. "Bernie Sanders".
        speaker_parties: All distinct political parties of speakers, e.g. "R".
        chambers: All distinct chambers in which speeches were given, e.g. "House".
        congresses: All distinct numbers of the congresses in which speeches were given, e.g. 114.
    """

    full_date_range: ClassVar[tuple[str, str]] = ("1996-01-01", "2016-06-30")
    speaker_names: ClassVar[set[str]] = {
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
    speaker_parties: ClassVar[set[str]] = {"D", "I", "R"}
    chambers: ClassVar[set[str]] = {"Extensions", "House", "Senate"}
    congresses: ClassVar[set[int]] = {
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        111,
        112,
        113,
        114,
    }

    def __init__(
        self,
        data_dir: types.PathLike = constants.DEFAULT_DATA_DIR.joinpath(NAME),
    ):
        super().__init__(NAME, meta=META)
        self.data_dir = utils.to_path(data_dir).resolve()
        self._filename = "capitol-words-py3.json.gz"
        self._filepath = self.data_dir.joinpath(self._filename)

    @property
    def filepath(self) -> Optional[str]:
        """
        Full path on disk for CapitolWords data as compressed json file.
        ``None`` if file is not found, e.g. has not yet been downloaded.
        """
        if self._filepath.is_file():
            return str(self._filepath)
        else:
            return None

    def download(self, *, force: bool = False) -> None:
        """
        Download the data as a Python version-specific compressed json file and
        save it to disk under the ``data_dir`` directory.

        Args:
            force: If True, download the dataset, even if it already exists
                on disk under ``data_dir``.
        """
        data_version = 1.0
        release_tag = f"capitol_words_py3_v{data_version}"
        url = urllib.parse.urljoin(DOWNLOAD_ROOT, release_tag + "/" + self._filename)
        tio.download_file(
            url, filename=self._filename, dirpath=self.data_dir, force=force
        )

    def __iter__(self):
        if not self._filepath.is_file():
            raise OSError(
                f"dataset file {self._filepath} not found;\n"
                "has the dataset been downloaded yet?"
            )
        for record in tio.read_json(self._filepath, mode="rt", lines=True):
            yield record

    def _get_filters(
        self,
        speaker_name: Optional[str | set[str]] = None,
        speaker_party: Optional[str | set[str]] = None,
        chamber: Optional[str | set[str]] = None,
        congress: Optional[int | set[int]] = None,
        date_range: Optional[tuple[Optional[str], Optional[str]]] = None,
        min_len: Optional[int] = None,
    ) -> list[Callable[[dict[str, Any]], bool]]:
        filters = []
        if min_len is not None:
            if min_len < 1:
                raise ValueError("`min_len` must be at least 1")
            min_len_ = min_len  # doing this so mypy stops complaining
            filters.append(lambda record: len(record.get("text", "")) >= min_len_)
        if date_range is not None:
            date_range_: tuple[str, str] = utils.validate_and_clip_range(
                date_range, self.full_date_range, val_type=(str, bytes)  # type: ignore
            )
            filters.append(
                lambda record: (
                    record.get("date")
                    and date_range_[0] <= record["date"] < date_range_[1]
                )
            )
        if speaker_name is not None:
            speaker_name_ = utils.validate_set_members(
                speaker_name, (str, bytes), valid_vals=self.speaker_names
            )
            filters.append(lambda record: record.get("speaker_name") in speaker_name_)
        if speaker_party is not None:
            speaker_party_ = utils.validate_set_members(
                speaker_party, (str, bytes), valid_vals=self.speaker_parties
            )
            filters.append(lambda record: record.get("speaker_party") in speaker_party_)
        if chamber is not None:
            chamber_ = utils.validate_set_members(
                chamber, (str, bytes), valid_vals=self.chambers
            )
            filters.append(lambda record: record.get("chamber") in chamber_)
        if congress is not None:
            congress_ = utils.validate_set_members(
                congress, int, valid_vals=self.congresses
            )
            filters.append(lambda record: record.get("congress") in congress_)
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
        speaker_name: Optional[str | set[str]] = None,
        speaker_party: Optional[str | set[str]] = None,
        chamber: Optional[str | set[str]] = None,
        congress: Optional[int | set[int]] = None,
        date_range: Optional[tuple[Optional[str], Optional[str]]] = None,
        min_len: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Iterable[str]:
        """
        Iterate over speeches in this dataset, optionally filtering by a variety
        of metadata and/or text length, and yield texts only,
        in chronological order.

        Args:
            speaker_name: Filter speeches by the speakers' name;
                see :attr:`CapitolWords.speaker_names`.
            speaker_party: Filter speeches by the speakers' party;
                see :attr:`CapitolWords.speaker_parties`.
            chamber: Filter speeches by the chamber in which they were given;
                see :attr:`CapitolWords.chambers`.
            congress: Filter speeches by the congress in which they were given;
                see :attr:`CapitolWords.congresses`.
            date_range: Filter speeches by the date on which they were given.
                Both start and end date must be specified, but a null value for either
                will be replaced by the min/max date available for the dataset.
            min_len: Filter texts by the length (# characters) of their text content.
            limit: Yield no more than ``limit`` texts that match all specified filters.

        Yields:
            Full text of next (by chronological order) speech in dataset
            passing all filter params.

        Raises:
            ValueError: If any filtering options are invalid.
        """
        filters = self._get_filters(
            speaker_name, speaker_party, chamber, congress, date_range, min_len
        )
        for record in itertools.islice(self._filtered_iter(filters), limit):
            yield record["text"]

    def records(
        self,
        *,
        speaker_name: Optional[str | set[str]] = None,
        speaker_party: Optional[str | set[str]] = None,
        chamber: Optional[str | set[str]] = None,
        congress: Optional[int | set[int]] = None,
        date_range: Optional[tuple[Optional[str], Optional[str]]] = None,
        min_len: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Iterable[types.Record]:
        """
        Iterate over speeches in this dataset, optionally filtering by a variety
        of metadata and/or text length, and yield text + metadata pairs,
        in chronological order.

        Args:
            speaker_name: Filter speeches by the speakers' name;
                see :attr:`CapitolWords.speaker_names`.
            speaker_party: Filter speeches by the speakers' party;
                see :attr:`CapitolWords.speaker_parties`.
            chamber: Filter speeches by the chamber in which they were given;
                see :attr:`CapitolWords.chambers`.
            congress: Filter speeches by the congress in which they were given;
                see :attr:`CapitolWords.congresses`.
            date_range: Filter speeches by the date on which they were given.
                Both start and end date must be specified, but a null value for either
                will be replaced by the min/max date available for the dataset.
            min_len: Filter speeches by the length (# characters) of their text content.
            limit: Yield no more than ``limit`` speeches that match all specified filters.

        Yields:
            Full text of the next (by chronological order) speech in dataset
            passing all filters, and its corresponding metadata.

        Raises:
            ValueError: If any filtering options are invalid.
        """
        filters = self._get_filters(
            speaker_name, speaker_party, chamber, congress, date_range, min_len
        )
        for record in itertools.islice(self._filtered_iter(filters), limit):
            yield types.Record(text=record.pop("text"), meta=record)
