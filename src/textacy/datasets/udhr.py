"""
UDHR translations
-----------------

A collection of translations of the Universal Declaration of Human Rights (UDHR),
a milestone document in the history of human rights that first, formally established
fundamental human rights to be universally protected.

Records include the following fields:

    - ``text``: Full text of the translated UDHR document.
    - ``lang``: ISO-639-1 language code of the text.
    - ``lang_name``: Ethnologue entry for the language (see https://www.ethnologue.com).

The source dataset was compiled and is updated by the Unicode Consortium
as a way to demonstrate the use of unicode in representing a wide variety of languages.
In fact, the UDHR was chosen because it's been translated into more languages
than any other document! However, this dataset only provides access to records
translated into ISO-639-1 languages — that is, major living languages *only*,
rather than every language, major or minor, that has ever existed. If you need access
to texts in those other languages, you can find them at :attr:`UDHR._texts_dirpath`.

For more details, go to https://unicode.org/udhr.
"""
from __future__ import annotations

import io
import itertools
import logging
import xml
from typing import Any, Iterable, Optional
from xml.etree import ElementTree

from .. import constants, preprocessing, types, utils
from .. import io as tio
from .base import Dataset

LOGGER = logging.getLogger(__name__)

NAME = "udhr"
META = {
    "site_url": "https://www.ohchr.org/en/human-rights/universal-declaration/universal-declaration-human-rights/about-universal-declaration-human-rights-translation-project",
    "description": (
        "A collection of translations of the Universal Declaration of Human Rights (UDHR), "
        "a milestone document in the history of human rights that first, formally established "
        "fundamental human rights to be universally protected."
    ),
}
DOWNLOAD_URL = "https://unicode.org/udhr/assemblies/udhr_txt.zip"


class UDHR(Dataset):
    """
    Stream a collection of UDHR translations from disk, either as texts or
    text + metadata pairs.

    Download the data (one time only!), saving and extracting its contents to disk::

        >>> import textacy.datasets
        >>> ds = textacy.datasets.UDHR()
        >>> ds.download()
        >>> ds.info
        {'name': 'udhr',
         'site_url': 'http://www.ohchr.org/EN/UDHR',
         'description': 'A collection of translations of the Universal Declaration of Human Rights (UDHR), a milestone document in the history of human rights that first, formally established fundamental human rights to be universally protected.'}

    Iterate over translations as texts or records with both text and metadata::

        >>> for text in ds.texts(limit=5):
        ...     print(text[:500])
        >>> for text, meta in ds.records(limit=5):
        ...     print("\\n{} ({})\\n{}".format(meta["lang_name"], meta["lang"], text[:500]))

    Filter translations by language, and note that some languages have multiple translations::

        >>> for text, meta in ds.records(lang="en"):
        ...     print("\\n{} ({})\\n{}".format(meta["lang_name"], meta["lang"], text[:500]))
        >>> for text, meta in ds.records(lang="zh"):
        ...     print("\\n{} ({})\\n{}".format(meta["lang_name"], meta["lang"], text[:500]))

    Note: Streaming translations into a :class:`textacy.Corpus <textacy.corpus.Corpus>`
    doesn't work as for other available datasets, since this dataset is multilingual.

    Args:
        data_dir (str or :class:`pathlib.Path`): Path to directory on disk
            under which the data is stored, i.e. ``/path/to/data_dir/udhr``.

    Attributes:
        langs (set[str]): All distinct language codes with texts in this dataset,
            e.g. "en" for English.
    """

    def __init__(
        self,
        data_dir: types.PathLike = constants.DEFAULT_DATA_DIR.joinpath(NAME),
    ):
        super().__init__(NAME, meta=META)
        self.data_dir = utils.to_path(data_dir).resolve()
        self._texts_dirpath = self.data_dir.joinpath("udhr_txt")
        self._index_filepath = self._texts_dirpath.joinpath("index.xml")
        self._index: Optional[list[dict[str, Any]]] = None
        self.langs: Optional[set[str]] = None

    def download(self, *, force: bool = False) -> None:
        """
        Download the data as a zipped archive of language-specific text files,
        then save it to disk and extract its contents under the ``data_dir`` directory.

        Args:
            force: If True, download the dataset, even if it already exists
                on disk under ``data_dir``.
        """
        filepath = tio.download_file(
            DOWNLOAD_URL, filename="udhr_txt.zip", dirpath=self.data_dir, force=force
        )
        if filepath:
            tio.unpack_archive(filepath, extract_dir=self.data_dir.joinpath("udhr_txt"))
        self._check_data()

    def _check_data(self):
        """Check that necessary data is found on disk, or raise an OSError."""
        if not self._texts_dirpath.is_dir():
            raise OSError(
                f"data directory {self._texts_dirpath} not found; "
                "has the dataset been downloaded?"
            )
        if not self._index_filepath.is_file():
            raise OSError(
                f"data index file {self._index_filepath} not found; "
                "has the dataset been downloaded?"
            )

    @property
    def index(self) -> Optional[list[dict[str, Any]]]:
        if not self._index:
            try:
                self._index = self._load_and_parse_index()
            except OSError as e:
                LOGGER.error(e)
        return self._index

    def _load_and_parse_index(self) -> list[dict[str, Any]]:
        """
        Read in index xml file from :attr:`UDHR._index_filepath`; skip elements
        without valid ISO-639-1 language code or sufficient translation quality,
        then convert into a list of dicts with key metadata, including filenames.
        """
        index: list[dict] = []
        tree = ElementTree.parse(self._index_filepath)
        root = tree.getroot()
        for ele in root.iterfind("udhr"):
            iso_lang_code = ele.get("bcp47", "").split("-", 1)[0]
            stage = int(ele.get("stage"))
            if len(iso_lang_code) != 2 or stage < 3:
                continue
            else:
                index.append(
                    {
                        "filename": f"udhr_{ele.get('f')}.txt",
                        "lang": iso_lang_code,
                        "lang_name": ele.get("n"),
                    }
                )
        # get set of all available langs, so users can filter on it
        self.langs = {item["lang"] for item in index}
        return index

    def _load_and_parse_text_file(self, filepath) -> str:
        with io.open(filepath, mode="rt", encoding="utf-8") as f:
            text_lines = [line.strip() for line in f.readlines()]
        # chop off the header, if it exists
        try:
            header_idx = text_lines.index("---")
            text_lines = text_lines[header_idx + 1 :]
        except ValueError:
            pass
        return preprocessing.normalize.whitespace("\n".join(text_lines))

    def __iter__(self):
        self._check_data()
        assert self.index is not None  # type guard
        for item in self.index:
            filepath = self._texts_dirpath.joinpath(item["filename"])
            record = item.copy()
            record["text"] = self._load_and_parse_text_file(filepath)
            yield record

    def _filtered_iter(self, lang):
        # this dataset is unusual in that the only filter we can really offer is lang
        # so we might as well avoid loading texts in unwanted languages
        if lang:
            self._check_data()
            assert self.index is not None  # type guard
            lang = utils.validate_set_members(lang, str, valid_vals=self.langs)
            for item in self.index:
                if item["lang"] in lang:
                    filepath = self._texts_dirpath.joinpath(item["filename"])
                    record = item.copy()
                    record["text"] = self._load_and_parse_text_file(filepath)
                    yield record
        else:
            for record in self:
                yield record

    def texts(
        self,
        *,
        lang: Optional[str | set[str]] = None,
        limit: Optional[int] = None,
    ) -> Iterable[str]:
        """
        Iterate over records in this dataset, optionally filtering by language,
        and yield texts only.

        Args:
            lang: Filter records by the language in which they're written;
                see :attr:`UDHR.langs`.
            limit: Yield no more than ``limit`` texts that match specified filter.

        Yields:
            Text of the next record in dataset passing filters.

        Raises:
            ValueError: If any filtering options are invalid.
        """
        for record in itertools.islice(self._filtered_iter(lang), limit):
            yield record["text"]

    def records(
        self,
        *,
        lang: Optional[str | set[str]] = None,
        limit: Optional[int] = None,
    ) -> Iterable[types.Record]:
        """
        Iterate over reocrds in this dataset, optionally filtering by a language,
        and yield text + metadata pairs.

        Args:
            lang: Filter records by the language in which they're written;
                see :attr:`UDHR.langs`.
            limit: Yield no more than ``limit`` texts that match specified filter.

        Yields:
            Text of the next record in dataset passing filters,
            and its corresponding metadata.

        Raises:
            ValueError: If any filtering options are invalid.
        """
        for record in itertools.islice(self._filtered_iter(lang), limit):
            yield types.Record(text=record.pop("text"), meta=record)
