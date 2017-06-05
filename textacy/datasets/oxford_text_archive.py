# -*- coding: utf-8 -*-
"""
Oxford Text Archive
-------------------

Description.

Record include the following key fields (plus a few others):

    * ``text``: desc
    * ``title``: desc
    * ``author``: desc
    * ``year``: desc
    * ``id``: desc
    * ``url``: desc
"""
from __future__ import unicode_literals

import io
import logging
import os
import re

import requests

from textacy import data_dir
from textacy import compat
from textacy.datasets.base import Dataset
from textacy import fileio
from textacy import preprocess

LOGGER = logging.getLogger(__name__)

NAME = 'oxford_text_archive'
DESCRIPTION = ('Collection of ~2.7k Creative Commons texts from the Oxford Text '
               'Archive, containing 16th-20th century literature and history.')
SITE_URL = 'https://ota.ox.ac.uk/'
DOWNLOAD_ROOT = 'https://github.com/mimno/ota/archive/master.zip'
DATA_DIR = os.path.join(data_dir, NAME)


class OxfordTextArchive(Dataset):
    """
    TODO
    """
    metadata = None

    def __init__(self, data_dir=DATA_DIR):
        super(OxfordTextArchive, self).__init__(
            name=NAME, description=DESCRIPTION, site_url=SITE_URL, data_dir=data_dir)
        self._filename = os.path.join(data_dir, 'ota-master')

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
        Download dataset from ``DOWNLOAD_ROOT`` and save it to disk under the
        :attr:`OxfordTextArchive.data_dir` directory.

        Args:
            force (bool): Download the file, even if it already exists on disk.
        """
        url = DOWNLOAD_ROOT
        fname = self._filename
        if os.path.isfile(fname) and force is False:
            LOGGER.warning(
                'File %s already exists; skipping download...', fname)
            return
        LOGGER.info(
            'Downloading data from %s and writing it to %s', url, fname)
        fileio.write_streaming_download_file(
            url, fname, mode='wb', encoding=None,
            auto_make_dirs=True, chunk_size=1024)

    def _load_and_parse_metadata(self):
        """
        """
        re_extract_year = re.compile(r'(\d{4})')
        re_extract_authors = re.compile(r'([^\d]+)(?:\d{4}(?:\?| or \d{1,2})?-(?:ca\. )?\d{4}|[bdfl]\.(?: ca\.)? \d{4}\??|-\d{4}|\d{4} or \d{1,2}|\d{2}th cent\.)\.?')
        re_clean_authors = re.compile(r'^[,; ]+|[,.]+\s*?$')

        metadata = []
        with ZipFile(self._filename, mode='r') as f:
            subf = io.StringIO(f.read('ota-master/metadata.tsv').decode('utf-8'))
            for row in csv.DictReader(subf, delimiter='\t'):
                # only include English-language works (99.9% of all works)
                if not row['Language'].startswith('English'):
                    continue
                # clean up years
                year_match = re_extract_year.search(row['Year'])
                if year_match:
                    row['Year'] = year_match.group()
                else:
                    row['Year'] = None
                # extract and clean up authors
                authors = re_extract_authors.findall(row['Author']) or [row['Author']]
                row['Author'] = [re_clean_authors.sub('', author) for author in authors]
                # get rid of uniform "Language" and "License" fields
                del row['Language']
                del row['License']
                metadata.append({key.lower(): val for key, val in row.items()})

        self.metadata = metadata
