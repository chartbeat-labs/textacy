"""
Oxford Text Archive literary works
----------------------------------

A collection of ~2.7k Creative Commons literary works from the Oxford Text Archive,
containing primarily English-language 16th-20th century literature and history.

Records include the following data:

    - ``text``: Full text of the literary work.
    - ``title``: Title of the literary work.
    - ``author``: Author(s) of the literary work.
    - ``year``: Year that the literary work was published.
    - ``url``: URL at which literary work can be found online via the OTA.
    - ``id``: Unique identifier of the literary work within the OTA.

This dataset was compiled by David Mimno from the Oxford Text Archive and
stored in his GitHub repo to avoid unnecessary scraping of the OTA site. It is
downloaded from that repo, and excluding some light cleaning of its metadata,
is reproduced exactly here.
"""
from __future__ import annotations

import csv
import io
import itertools
import logging
import os
import re
from typing import Any, ClassVar, Iterable, Optional

from .. import constants
from .. import io as tio
from .. import types, utils
from .base import Dataset


LOGGER = logging.getLogger(__name__)

NAME = "oxford_text_archive"
META = {
    "site_url": "https://ota.ox.ac.uk/",
    "description": (
        "Collection of ~2.7k Creative Commons texts from the Oxford Text "
        "Archive, containing primarily English-language 16th-20th century "
        "literature and history."
    ),
}
DOWNLOAD_URL = "https://github.com/mimno/ota/archive/master.zip"


class OxfordTextArchive(Dataset):
    """
    Stream a collection of English-language literary works from text files on disk,
    either as texts or text + metadata pairs.

    Download the data (one time only!), saving and extracting its contents to disk::

        >>> import textacy.datasets
        >>> ds = textacy.datasets.OxfordTextArchive()
        >>> ds.download()
        >>> ds.info
        {'name': 'oxford_text_archive',
         'site_url': 'https://ota.ox.ac.uk/',
         'description': 'Collection of ~2.7k Creative Commons texts from the Oxford Text Archive, containing primarily English-language 16th-20th century literature and history.'}

    Iterate over literary works as texts or records with both text and metadata::

        >>> for text in ds.texts(limit=3):
        ...     print(text[:200])
        >>> for text, meta in ds.records(limit=3):
        ...     print("\\n{}, {}".format(meta["title"], meta["year"]))
        ...     print(text[:300])

    Filter literary works by a variety of metadata fields and text length::

        >>> for text, meta in ds.records(author="Shakespeare, William", limit=1):
        ...     print("{}\\n{}".format(meta["title"], text[:500]))
        >>> for text, meta in ds.records(date_range=("1900-01-01", "1990-01-01"), limit=5):
        ...     print(meta["year"], meta["author"])
        >>> for text in ds.texts(min_len=4000000):
        ...     print(len(text))

    Stream literary works into a :class:`textacy.Corpus <textacy.corpus.Corpus>`::

        >>> textacy.Corpus("en", data=ds.records(limit=5))
        Corpus(5 docs; 182289 tokens)

    Args:
        data_dir (str or :class:`pathlib.Path`): Path to directory on disk
            under which dataset is stored, i.e. ``/path/to/data_dir/oxford_text_archive``.

    Attributes:
        full_date_range: First and last dates for which works are available,
            each as an ISO-formatted string (YYYY-MM-DD).
        authors (set[str]): Full names of all distinct authors included in this
            dataset, e.g. "Shakespeare, William".
    """

    full_date_range: ClassVar[tuple[str, str]] = ("0018-01-01", "1990-01-01")

    def __init__(
        self,
        data_dir: types.PathLike = constants.DEFAULT_DATA_DIR.joinpath(NAME),
    ):
        super().__init__(NAME, meta=META)
        self.data_dir = utils.to_path(data_dir).resolve()
        self._text_dirpath = self.data_dir.joinpath("master", "text")
        self._metadata_filepath = self.data_dir.joinpath("master", "metadata.tsv")
        self._metadata: Optional[dict[str, dict[str, Any]]] = None

    def download(self, *, force: bool = False) -> None:
        """
        Download the data as a zip archive file, then save it to disk and
        extract its contents under the :attr:`OxfordTextArchive.data_dir` directory.

        Args:
            force: If True, download the dataset, even if it already exists
                on disk under ``data_dir``.
        """
        filepath = tio.download_file(
            DOWNLOAD_URL, filename=None, dirpath=self.data_dir, force=force
        )
        if filepath:
            tio.unpack_archive(filepath, extract_dir=None)

    @property
    def metadata(self) -> Optional[dict[str, dict[str, Any]]]:
        if not self._metadata:
            try:
                self._metadata = self._load_and_parse_metadata()
            except OSError as e:
                LOGGER.error(e)
        return self._metadata

    def _load_and_parse_metadata(self) -> dict[str, dict[str, Any]]:
        """
        Read in ``metadata.tsv`` file from :attr:`OxfordTextArchive._metadata_filepath``
        zip archive; convert into a dictionary keyed by record ID; clean up some
        of the fields, and remove a couple fields that are identical throughout.
        """
        if not self._metadata_filepath.is_file():
            raise OSError(
                f"metadata file {self._metadata_filepath} not found;\n"
                "has the dataset been downloaded yet?"
            )

        re_extract_year = re.compile(r"(\d{4})")
        re_extract_authors = re.compile(
            r"(\D+)"
            r"(?:, "
            r"(?:[bdf]l?\. )?(?:ca. )?\d{4}(?:\?| or \d{1,2})?(?:-(?:[bdf]l?\. )?(?:ca. )?\d{4}(?:\?| or \d{1,2})?)?|"
            r"(?:\d{2}th(?:/\d{2}th)? cent\.)"
            r"\.?)"
        )
        re_clean_authors = re.compile(r"^[,;. ]+|[,.]+\s*?$")
        metadata = {}
        with self._metadata_filepath.open(mode="rb") as f:
            subf = io.StringIO(f.read().decode("utf-8"))
            for row in csv.DictReader(subf, delimiter="\t"):
                # only include English-language works (99.9% of all works)
                if not row["Language"].startswith("English"):
                    continue
                # clean up years
                year_match = re_extract_year.search(row["Year"])
                if year_match:
                    row["Year"] = year_match.group()
                else:
                    row["Year"] = None
                # extract and clean up authors
                authors = re_extract_authors.findall(row["Author"]) or [row["Author"]]
                row["Author"] = tuple(
                    re_clean_authors.sub("", author) for author in authors
                )
                row["Title"] = row["Title"].strip()
                # get rid of uniform "Language" and "License" fields
                del row["Language"]
                del row["License"]
                metadata[row["ID"]] = {key.lower(): val for key, val in row.items()}
        # set authors attribute for user convenience / to validate author filtering
        self.authors = {
            author
            for value in metadata.values()
            for author in value["author"]
            if value.get("author")
        }
        return metadata

    def __iter__(self):
        if not self._text_dirpath.is_dir():
            raise OSError(
                f"text directory {self._text_dirpath} not found;\n"
                "has the dataset been downloaded yet?"
            )
        _metadata = self.metadata  # for performance
        for filepath in sorted(tio.get_filepaths(self._text_dirpath, extension=".txt")):
            id_, _ = os.path.splitext(os.path.basename(filepath))
            record = _metadata.get(id_, {}).copy()
            if not record:
                LOGGER.debug(
                    "no metadata found for record %s; probably non-English text...", id_
                )
                continue
            with io.open(filepath, mode="rt", encoding="utf-8") as f:
                record["text"] = f.read()
            yield record

    def _get_filters(self, author, date_range, min_len):
        filters = []
        if min_len is not None:
            if min_len < 1:
                raise ValueError("`min_len` must be at least 1")
            filters.append(lambda record: len(record.get("text", "")) >= min_len)
        if author is not None:
            author = utils.validate_set_members(
                author, (str, bytes), valid_vals=self.authors
            )
            filters.append(
                lambda record: record.get("author")
                and any(athr in author for athr in record["author"])
            )
        if date_range is not None:
            date_range = utils.validate_and_clip_range(
                date_range, self.full_date_range, val_type=(str, bytes)
            )
            filters.append(
                lambda record: record.get("year")
                and date_range[0] <= record["year"] < date_range[1]
            )
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
        author: Optional[str | set[str]] = None,
        date_range: Optional[tuple[Optional[str], Optional[str]]] = None,
        min_len: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Iterable[str]:
        """
        Iterate over works in this dataset, optionally filtering by a variety
        of metadata and/or text length, and yield texts only.

        Args:
            author: Filter texts by the authors' name. For multiple values (set[str]),
                ANY rather than ALL of the authors must be found among a given works's authors.
            date_range: Filter texts by the date on which it was published;
                both start and end date must be specified, but a null value for either
                will be replaced by the min/max date available in the dataset.
            min_len: Filter texts by the length (# characters) of their text content.
            limit: Yield no more than ``limit`` texts that match all specified filters.

        Yields:
            Text of the next work in dataset passing all filters.

        Raises:
            ValueError: If any filtering options are invalid.
        """
        filters = self._get_filters(author, date_range, min_len)
        for record in itertools.islice(self._filtered_iter(filters), limit):
            yield record["text"]

    def records(
        self,
        *,
        author: Optional[str | set[str]] = None,
        date_range: Optional[tuple[Optional[str], Optional[str]]] = None,
        min_len: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Iterable[types.Record]:
        """
        Iterate over works in this dataset, optionally filtering by a variety
        of metadata and/or text length, and yield text + metadata pairs.

        Args:
            author: Filter texts by the authors' name. For multiple values (set[str]),
                ANY rather than ALL of the authors must be found among a given works's authors.
            date_range: Filter texts by the date on which it was published;
                both start and end date must be specified, but a null value for either
                will be replaced by the min/max date available in the dataset.
            min_len: Filter texts by the length (# characters) of their text content.
            limit: Yield no more than ``limit`` texts that match all specified filters.

        Yields:
            Text of the next work in dataset passing all filters,
            and its corresponding metadata.

        Raises:
            ValueError: If any filtering options are invalid.
        """
        filters = self._get_filters(author, date_range, min_len)
        for record in itertools.islice(self._filtered_iter(filters), limit):
            yield types.Record(text=record.pop("text"), meta=record)
