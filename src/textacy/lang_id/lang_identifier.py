"""
Language Identification
-----------------------

:mod:`textacy.lang_id`: Interface for de/serializing a language identification model,
and using it to identify the most probable language(s) of a given text. Inspired by
-- and using the same methodology as -- Facebook's fastText
(https://fasttext.cc/blog/2017/10/02/blog-post.html).

Model
^^^^^

Text is tokenized into a bag of word 1- and 2-grams and character 1- through 5-grams.
The collection of n-grams is embedded into a 128-dimensional space, then averaged.
The resulting features are fed into a linear classifier with a hierarchical softmax output
to compute (approximate) language probabilities for 140 ISO 639-1 languages.

Dataset
^^^^^^^

The model was trained on a randomized, stratified subset of ~2.9M texts
drawn from several sources:

- **WiLi:** A public dataset of short text extracts from Wikipedias in over 230
  languages. Style is relatively formal; subject matter is "encyclopedic".
  Source: https://zenodo.org/record/841984
- **Tatoeba:** A crowd-sourced collection of sentences and their translations into
  many languages. Style is relatively informal; subject matter is a variety of
  everyday things and goings-on.
  Source: https://tatoeba.org/eng/downloads.
- **UDHR:** The UN's Universal Declaration of Human Rights document, translated into
  hundreds of languages and split into paragraphs. Style is formal; subject matter is
  fundamental human rights to be universally protected.
  Source: https://unicode.org/udhr/index.html
- **DSLCC**: Two collections of short excerpts of journalistic texts in a handful
  of language groups that are highly similar to each other. Style is relatively formal;
  subject matter is current events.
  Source: http://ttg.uni-saarland.de/resources/DSLCC/
- **Ted 2020**: A crawl of nearly 4000 TED and TED-X transcripts from 2020,
   translated by a global community of volunteers into more than 100 languages.
   Style is conversational, covering a broad range of subjects.
   Source: https://opus.nlpl.eu/TED2020.php
- **SETimes**: A corpus of news articles in Balkan languages, originally extracted
  from http://www.setimes.com and compiled by Nikola Ljubešić.
  Source: https://opus.nlpl.eu/SETIMES.php

Performance
^^^^^^^^^^^

The trained model achieved F1 = 0.97 when averaged over all languages.

A few languages have worse performance; most notably, the two sub-Norwegians ("nb" and "no"),
as well as Bosnian ("bs"), Serbian ("sr"), and Croatian ("hr"), which are extremely
similar to each other. See the textacy-data releases for more details:
https://github.com/bdewilde/textacy-data/releases/tag/lang-identifier-v3.0
"""
from __future__ import annotations

import logging
import pathlib
import urllib.parse

import floret
from floret.floret import _floret

from .. import utils
from ..constants import DEFAULT_DATA_DIR


LOGGER = logging.getLogger(__name__)


class LangIdentifier:
    """
    Args:
        version
        data_dir

    Attributes:
        model
        classes
    """

    def __init__(
        self,
        version: str = "3.0",
        data_dir: str | pathlib.Path = DEFAULT_DATA_DIR.joinpath("lang_identifier"),
    ):
        self.data_dir = utils.to_path(data_dir)
        self.version = version
        self._model = None
        self._classes = None
        self._label_prefix = "__label__"

    @property
    def model_id(self) -> str:
        return f"lang-identifier-v{self.version}"

    @property
    def model_fpath(self) -> pathlib.Path:
        return self.data_dir.joinpath(f"{self.model_id}.bin")

    @property
    def model(self) -> _floret:
        if self._model is None:
            self._model = floret.load_model(str(self.model_fpath))
            if hasattr(self._model, "label"):
                self._label_prefix = self._model.label
        return self._model

    @property
    def classes(self) -> list[str]:
        if self._classes is None:
            labels = self.model.labels
            assert isinstance(labels, list)  # type guard
            self._classes = sorted(self._to_lang(label) for label in labels)
        return self._classes

    def _to_lang(self, label: str) -> str:
        return label.removeprefix(self._label_prefix)

    def save_model(self):
        """Save trained :attr:`LangIdentifier.model` to disk."""
        LOGGER.info("saving LangIdentifier model to %s", self.model_fpath)
        self.model.save_model(str(self.model_fpath))

    def load_model(self) -> _floret:
        """Load trained model from disk."""
        try:
            LOGGER.debug("loading LangIdentifier model from %s", self.model_fpath)
            return floret.load_model(str(self.model_fpath))
        except FileNotFoundError:
            LOGGER.exception(
                "LangIdentifier model not found at %s -- have you downloaded it yet?",
                self.model_fpath,
            )
            raise

    def download(self, force: bool = False):
        """
        Download version-specific model data as a binary file and save it to disk
        at :attr:`LangIdentifier.model_fpath`.

        Args:
            force: If True, download the model data, even if it already exists on disk
                under :attr:`self.data_dir`; otherwise, don't.
        """
        # hide this import, since we'll only ever need it _once_ (per model version)
        from .. import io as tio

        model_fname = self.model_fpath.name
        url = urllib.parse.urljoin(
            "https://github.com/bdewilde/textacy-data/releases/download/",
            self.model_id + "/" + model_fname,
        )
        tio.utils.download_file(
            url, filename=model_fname, dirpath=self.data_dir, force=force
        )

    def identify_lang(
        self, text: str, with_probs: bool = False
    ) -> str | tuple[str, float]:
        """
        Identify the most probable language identified in ``text``,
        with or without the corresponding probability.

        Args:
            text
            with_probs

        Returns:
            ISO 639-1 standard language code of the most probable language,
            optionally with its probability.
        """
        if not self._is_valid_text(text):
            result = ("un", 1.0)
        else:
            result_ = self.model.predict(text, k=1)
            result: tuple[str, float] = (
                self._to_lang(result_[0][0]),  # type: ignore
                float(result_[1][0]),
            )
        return result[0] if with_probs is False else result

    def identify_topn_langs(
        self,
        text: str,
        topn: int = 3,
        with_probs: bool = False,
    ) -> list[str] | list[tuple[str, float]]:
        """
        Identify the ``topn`` most probable languages identified in ``text``,
        with or without the corresponding probabilities.

        Args:
            text
            topn
            with_probs

        Returns:
            ISO 639-1 standard language code, optionally with its probability,
            of the ``topn`` most probable languages.
        """
        if not self._is_valid_text(text):
            results = [("un", 1.0)]
        else:
            results_ = self.model.predict(text, k=topn)
            results: list[tuple[str, float]] = [
                (self._to_lang(result[0]), float(result[1]))
                for result in zip(results_[0], results_[1])
            ]
        return [lang for lang, _ in results] if with_probs is False else results

    def _is_valid_text(self, text: str) -> bool:
        return any(char.isalpha() for char in text)


lang_identifier = LangIdentifier(
    version="3.0", data_dir=DEFAULT_DATA_DIR.joinpath("lang_identifier")
)
# expose this as primary user-facing API
# TODO: there's gotta be a better way, this whole setup feels clunky
identify_lang = lang_identifier.identify_lang
identify_topn_langs = lang_identifier.identify_topn_langs
