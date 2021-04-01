from __future__ import annotations

import logging
import operator
import pathlib
from typing import Dict, List, Optional, Tuple, Set

from cytoolz import itertoolz

import textacy

LOGGER = logging.getLogger(__name__)


class IsoLangResource:
    """
    Dataset based on the official ISO-639 code table,
    mapping all language code variations (639-1, 639-2, 639-3) to each other.

    Source: https://iso639-3.sil.org/code_tables/639/data
    """

    download_url = "https://iso639-3.sil.org/sites/iso639-3/files/downloads/iso-639-3.tab"
    filename = "iso-639-3.tsv"

    def __init__(self, data_dir: str | pathlib.Path):
        self.data_dir = textacy.utils.to_path(data_dir).resolve()

    def download(self, force: bool = False):
        """
        Args:
            force: If True, always download a new copy of the dataset; otherwise,
                only download dataset if it doesn't already exist on disk.
        """
        textacy.io.download_file(
            self.download_url,
            filename=self.filename,
            dirpath=self.data_dir,
            force=force,
        )

    def load(self, exclude: Optional[Set[str]] = None) -> Dict[str, str]:
        """
        Args:
            exclude

        Returns:
            Dict[str, str]
        """
        rows = textacy.io.read_csv(
            self.data_dir.joinpath(self.filename),
            delimiter="\t",
            fieldnames=[
                "Id",
                "Part2B",
                "Part2T",
                "Part1",
                "Scope",
                "Language_Type",
                "Ref_Name",
                "Comment",
            ],
            quoting=1,
        )
        lang_map = {
            row["Id"]: row["Part1"]
            for row in rows
            if row.get("Part1") and (exclude is None or row["Part1"] not in exclude)
        }
        LOGGER.info(
            "loaded IsoLangResource data:\n%s ...",
            sorted(lang_map.items())[:5],
        )
        return lang_map


class DSLCCDataset:
    """
    Dataset based on two multilingual collections of short excerpts of journalistic texts,
    focused on language groups that are very similar and thus more difficult
    to correctly identify.

    Source: http://ttg.uni-saarland.de/resources/DSLCC
    """

    def __init__(self, data_dir: str | pathlib.Path):
        self.data_dir = textacy.utils.to_path(data_dir).resolve()

    def download(self, force: bool = False):
        """
        Args:
            force: If True, always download a new copy of the dataset; otherwise,
                only download dataset if it doesn't already exist on disk.
        """
        for version in [3, 4]:
            name = f"dslcc{version}"
            url = f"http://scholar.harvard.edu/files/malmasi/files/{name}.zip"
            fpath = textacy.io.download_file(url, dirpath=self.data_dir, force=force)
            if fpath:
                textacy.io.unpack_archive(fpath, extract_dir=self.data_dir.joinpath(name))

    def load(self, langs: Set[str], min_len: int = 25) -> List[Tuple[str, str]]:
        """
        Args:
            langs
            min_len: Minimum text length in *chars* for a given example to be included.

        Returns:
            Sequence of (text, lang) examples.
        """
        data = []
        fstubs = [
            "dslcc3/train/task1-train.txt",
            "dslcc3/train/task1-dev.txt",
            "dslcc4/DSL-TRAIN.txt",
            "dslcc4/DSL-DEV.txt",
        ]
        for fstub in fstubs:
            filepath = self.data_dir.joinpath(fstub)
            lines = textacy.io.read_text(filepath, mode="rt", encoding="utf-8", lines=True)
            for line in lines:
                if not line.strip():
                    continue
                try:
                    text, lang = line.split("\t")
                    if (
                        lang[:2] in langs and
                        itertoolz.count(char for char in text if char.isalnum()) >= min_len
                    ):
                        data.append((text, lang[:2]))
                except Exception:
                    LOGGER.debug("bad line in data")
                    pass
        data = sorted(set(data), key=operator.itemgetter(1))
        LOGGER.info("loaded DSLCCDataset data:\n%s ...", data[:3])
        return data


class TatoebaDataset:

    download_url = "http://downloads.tatoeba.org/exports/sentences.tar.bz2"

    def __init__(self, data_dir: str | pathlib.Path):
        self.data_dir = textacy.utils.to_path(data_dir).resolve()

    def download(self, force: bool =False):
        """
        Args:
            force: If True, always download a new copy of the dataset; otherwise,
                only download dataset if it doesn't already exist on disk.

        Note:
            Source: https://tatoeba.org/eng/downloads
        """
        fpath = textacy.io.download_file(
            self.download_url, dirpath=self.data_dir, force=force
        )
        if fpath:
            textacy.io.unpack_archive(fpath, extract_dir=self.data_dir)

    def load(
        self,
        iso_lang_map: Dict[str, str],
        min_len: int = 25,
    ) -> List[Tuple[str, str]]:
        """
        Args:
            iso_lang_map
            min_len: Minimum text length in *chars* for a given example to be included.

        Returns:
            Sequence of (text, lang) examples.
        """
        rows = textacy.io.read_csv(
            self.data_dir.joinpath("sentences.csv"),
            fieldnames=["sent_id", "iso-639-3", "text"],
            delimiter="\t",
            quoting=1,
        )
        data = [
            (row["text"], iso_lang_map[row["iso-639-3"]])
            for row in rows
            if row["iso-639-3"] in iso_lang_map
            and itertoolz.count(char for char in row["text"] if char.isalnum()) >= min_len
        ]
        LOGGER.info("loaded TatoebaDataset data:\n%s ...", data[:3])
        return data


class Wili2018Dataset:
    """
    Dataset based on paragraphs from Wikipedia in 230+ languages.

    Source: https://zenodo.org/record/841984

    References:
        Thoma, Martin. "The WiLI benchmark dataset for written language identification."
        arXiv preprint arXiv:1801.07779 (2018).
    """

    download_url = "https://zenodo.org/record/841984/files/wili-2018.zip?download=1"

    def __init__(self, data_dir: str | pathlib.Path):
        self.data_dir = textacy.utils.to_path(data_dir).resolve()

    def download(self, force: bool = False):
        """
        Args:
            force: If True, always download a new copy of the dataset; otherwise,
                only download dataset if it doesn't already exist on disk.
        """
        fpath = textacy.io.download_file(
            self.download_url, dirpath=self.data_dir, force=force
        )
        if fpath:
            textacy.io.unpack_archive(fpath, extract_dir=self.data_dir)

    def load(
        self,
        iso_lang_map: Dict[str, str],
        min_len: int = 25,
    ) -> List[Tuple[str, str]]:
        """
        Args:
            iso_lang_map
            min_len: Minimum text length in *chars* for a given example to be included.

        Returns:
            Sequence of (text, lang) examples.
        """
        data = []
        # we'll combine train/test from individual datasets
        # and instead split on the full, aggregated dataset
        for subset in ("train", "test"):
            text_lines = textacy.io.read_text(
                self.data_dir.joinpath(f"x_{subset}.txt"), lines=True
            )
            lang_lines = textacy.io.read_text(
                self.data_dir.joinpath(f"y_{subset}.txt"), lines=True
            )
            texts = (line.strip() for line in text_lines)
            langs = (line.strip() for line in lang_lines)
            data.extend(
                (text, iso_lang_map[lang])
                for text, lang in zip(texts, langs)
                if lang in iso_lang_map
                and itertoolz.count(char for char in text if char.isalnum()) >= min_len
            )
        LOGGER.info("loaded Wili2018Dataset data:\n%s ...", data[:3])
        return data
