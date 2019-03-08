# -*- coding: utf-8 -*-
"""
Oxford Text Archive
-------------------

A collection of ~2.7k Creative Commons texts from the Oxford Text Archive,
containing primarily English-language 16th-20th century literature and history.

Record include the following fields:

    * ``text``: Full text of the literary work.
    * ``title``: Title of the literary work.
    * ``author``: Author(s) of the literary work.
    * ``year``: Year that the literary work was published.
    * ``url``: URL at which literary work can be found online via the OTA.

This dataset was compiled by [DAVID?] Mimno from the Oxford Text Archive and
stored in his GitHub repo to avoid unnecessary scraping of the OTA site. It is
downloaded from that repo, and excluding some light cleaning of its metadata,
is reproduced exactly here.
"""
from __future__ import unicode_literals

import logging
import os
import re
import zipfile
from io import StringIO

import requests

from .. import compat
from .. import data_dir
from .. import io
from .. import preprocess
from .base import Dataset

LOGGER = logging.getLogger(__name__)

NAME = "oxford_text_archive"
DESCRIPTION = (
    "Collection of ~2.7k Creative Commons texts from the Oxford Text "
    "Archive, containing primarily English-language 16th-20th century "
    "literature and history."
)
SITE_URL = "https://ota.ox.ac.uk/"
DOWNLOAD_ROOT = "https://github.com/mimno/ota/archive/master.zip"
DATA_DIR = os.path.join(data_dir, NAME)


class OxfordTextArchive(Dataset):
    """
    Stream literary works from a zip file on disk, either as texts (str) or
    records (dict) with both text content and metadata.

    Download the zip file from its GitHub repository::

        >>> ota = OxfordTextArchive()
        >>> ota.download()
        >>> ota.info
        {'data_dir': 'path/to/textacy/data/oxford_text_archive',
         'description': 'Collection of ~2.7k Creative Commons texts from the Oxford Text Archive, containing primarily English-language 16th-20th century literature and history.',
         'name': 'oxford_text_archive',
         'site_url': 'https://ota.ox.ac.uk/'}

    Iterate over literary works as plain texts or records with both text and metadata::

        >>> for text in ota.texts(limit=5):
        ...     print(text[:400])
        >>> for record in ota.records(limit=5):
        ...     print(record['title'], record['year'])
        ...     print(record['text'][:400])

    Filter literary works by a variety of metadata fields and text length::

        >>> for record in ota.records(author='Shakespeare, William', limit=1):
        ...     print(record['year'], record['text'])
        >>> for record in ota.records(date_range=('1900-01-01', '2000-01-01'), limit=5):
        ...     print(record['year'], record['author'])
        >>> for text in ota.texts(min_len=4000000):
        ...     print(len(text))
        ...     print(text[:200], '...')

    Stream literary works into a :class:`textacy.Corpus`::

        >>> text_stream, metadata_stream = textacy.io.split_records(
        ...     ota.records(limit=10), 'text')
        >>> c = textacy.Corpus('en', texts=text_stream, metadatas=metadata_stream)
        >>> c
        Corpus(10 docs; 686881 tokens)

    Args:
        data_dir (str): Path to directory on disk under which dataset's file
            ("ota-master.zip") is stored.

    Attributes:
        min_date (str): Earliest date for which speeches are available, as an
            ISO-formatted string ("YYYY-MM-DD").
        max_date (str): Latest date for which speeches are available, as an
            ISO-formatted string ("YYYY-MM-DD").
        authors (Set[str]): Full names of all distinct authors included in this
            dataset, e.g. "Shakespeare, William".
    """

    min_date = "0018-01-01"
    max_date = "1990-01-01"

    def __init__(self, data_dir=DATA_DIR):
        super(OxfordTextArchive, self).__init__(
            name=NAME, description=DESCRIPTION, site_url=SITE_URL, data_dir=data_dir
        )
        self._filename = os.path.join(data_dir, "ota-master.zip")
        try:
            self._metadata = self._load_and_parse_metadata()
        except IOError:
            pass

    @property
    def filename(self):
        """
        str: Full path on disk for OxfordTextArchive data as a zip archive file.
            ``None`` if file is not found, e.g. has not yet been downloaded.
        """
        if os.path.isfile(self._filename):
            return self._filename
        else:
            return None

    def download(self, force=False):
        """
        Download the data as a zip archive file and save it to disk under the
        ``data_dir`` directory.

        Args:
            force (bool): If True, download the dataset, even if it already
                exists on disk under ``data_dir``.
        """
        url = DOWNLOAD_ROOT
        fname = self._filename
        if os.path.isfile(fname) and force is False:
            LOGGER.warning("File %s already exists; skipping download...", fname)
            return
        LOGGER.info("Downloading data from %s and writing it to %s", url, fname)
        io.write_http_stream(
            url, fname, mode="wb", encoding=None, make_dirs=True, chunk_size=1024
        )
        self._metadata = self._load_and_parse_metadata()

    def _load_and_parse_metadata(self):
        """
        Read in ``metadata.tsv`` file from the :attr:`OxfordTextArchive.filename``
        zip archive; convert into a dictionary keyed by record ID; clean up some
        of the fields, and remove a couple fields that are identical throughout.
        """
        if not self.filename:
            raise IOError("{} file not found".format(self._filename))

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
        with zipfile.ZipFile(self._filename, mode="r") as f:
            subf = StringIO(f.read("ota-master/metadata.tsv").decode("utf-8"))
            for row in compat.csv.DictReader(subf, delimiter="\t"):
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
                row["Author"] = [re_clean_authors.sub("", author) for author in authors]
                # get rid of uniform "Language" and "License" fields
                del row["Language"]
                del row["License"]
                id_ = row.pop("ID")
                metadata[id_] = {key.lower(): val for key, val in row.items()}

        # set authors attribute
        self.authors = {
            author
            for value in metadata.values()
            for author in value["author"]
            if value.get("author")
        }

        return metadata

    def texts(self, author=None, date_range=None, min_len=None, limit=-1):
        """
        Iterate over works (text-only) in this dataset, optionally filtering
        by a variety of metadata and/or text length.

        Args:
            author (str or Set[str]): Filter texts by the authors' name;
                see :attr:`authors <OxfordTextArchive.authors>`.
            date_range (List[str] or Tuple[str]): Filter texts by the date on
                which it was published; both start and end date must be specified,
                but a null value for either will be replaced by the min/max date
                available in the dataset.
            min_len (int): Filter texts by the length (number of characters)
                of their text content.
            limit (int): Return no more than ``limit`` texts.

        Yields:
            str: Full text of next work in dataset passing all filter params.

        Raises:
            ValueError: If any filtering options are invalid.
        """
        texts = self._iterate(True, author, date_range, min_len, limit)
        for text in texts:
            yield text

    def records(self, author=None, date_range=None, min_len=None, limit=-1):
        """
        Iterate over works (including text and metadata) in this dataset,
        optionally filtering by a variety of metadata and/or text length.

        Args:
            author (str or Set[str]): Filter records by the authors' name;
                see :attr:`authors <OxfordTextArchive.authors>`.
            date_range (List[str] or Tuple[str]): Filter records by the date on
                which it was published; both start and end date must be specified,
                but a null value for either will be replaced by the min/max date
                available in the dataset.
            min_len (int): Filter records by the length (number of characters)
                of their text content.
            limit (int): Yield no more than ``limit`` records.

        Yields:
            dict: Text and metadata of next work in dataset passing all
            filter params.

        Raises:
            ValueError: If any filtering options are invalid.
        """
        records = self._iterate(False, author, date_range, min_len, limit)
        for record in records:
            yield record

    def _iterate(self, text_only, author, date_range, min_len, limit):
        """
        Low-level method to iterate over the records in this dataset. Used by
        :meth:`OxfordTextArchive.texts()` and :meth:`OxfordTextArchive.records()`.
        """
        if not self.filename:
            raise IOError("{} file not found".format(self._filename))

        if author:
            if isinstance(author, compat.string_types):
                author = {author}
            if not all(item in self.authors for item in author):
                raise ValueError(
                    "all values in `author` must be valid; "
                    "see :attr:`CapitolWords.authors`"
                )
        if date_range:
            date_range = self._parse_date_range(date_range)

        n = 0
        with zipfile.ZipFile(self.filename, mode="r") as f:
            for name in f.namelist():

                # other stuff in zip archive that isn't a text file
                if not name.startswith("ota-master/text/"):
                    continue
                id_ = os.path.splitext(os.path.split(name)[-1])[0]
                meta = self._metadata.get(id_)
                # this is actually just the "ota-master/text/" directory
                if not meta:
                    continue
                # filter by metadata
                if date_range:
                    if (
                        not meta.get("year")
                        or not date_range[0] <= meta["year"] <= date_range[1]
                    ):
                        continue
                if author:
                    if not meta.get("author") or not any(
                        a in author for a in meta["author"]
                    ):
                        continue
                text = f.read(name).decode("utf-8")
                if min_len and len(text) < min_len:
                    continue

                if text_only is True:
                    yield text
                else:
                    meta["text"] = text
                    yield meta

                n += 1
                if n == limit:
                    break
