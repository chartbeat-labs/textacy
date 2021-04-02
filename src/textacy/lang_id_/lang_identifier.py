from __future__ import annotations

import logging
import pathlib
import urllib
from typing import List, Tuple

from thinc.api import Model

from . import models
from .. import constants, utils


LOGGER = logging.getLogger(__name__)


class LangIdentifier:
    """
    Args:
        model_base
        data_dir
        version
    """

    def __init__(
        self,
        model_base: Model,
        data_dir: str | pathlib.Path,
        version: float | str,
    ):
        self.data_dir = utils.to_path(data_dir)
        self.version = str(version)
        self._model_base = model_base
        self._model = None
        self._classes = None

    @property
    def model_id(self) -> str:
        return f"lang-identifier-v{self.version}"

    @property
    def model_fpath(self) -> pathlib.Path:
        return self.data_dir.joinpath(f"{self.model_id}.bin")

    @property
    def model(self) -> Model:
        if self._model is None:
            self._model = self.load_model()
        return self._model

    @property
    def classes(self):
        if self._classes is None:
            self._classes = self.model.layers[-1].attrs["classes"]
        return self._classes

    def save_model(self):
        LOGGER.info("saving LangIdentifier model to %s", self.model_fpath)
        self.model.to_disk(self.model_fpath)

    def load_model(self):
        try:
            LOGGER.debug("loading LangIdentifier model from %s", self.model_fpath)
            self._model_base.from_disk(self.model_fpath)
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

        release_tag = self.model_id.replace("-", "_")
        model_fname = self.model_fpath.name
        url = urllib.parse.urljoin(
            "https://github.com/bdewilde/textacy-data/releases/download/",
            release_tag + "/" + model_fname,
        )
        tio.utils.download_file(
            url, filename=model_fname, dirpath=self.data_dir, force=force,
        )

    def identify_lang(
        self,
        text: str,
        with_probs: bool = False,
    ) -> str | Tuple[str, float]:
        """
        Identify the most probable language identified in ``text``.

        Args:
            text
            with_probs

        Returns:
            ISO 639-1 standard language code of the most probable language.
        """
        if not self._is_valid_text(text):
            result = ("un", 1.0)
        else:
            text_ = utils.to_collection(text, str, list)
            result = models.get_topn_preds_and_probs(
                self.model.predict(text_), 1, self.classes
            )[0]
        return result[0] if with_probs is False else result

    def identify_topn_langs(
        self,
        text: str,
        topn: int = 3,
        with_probs: bool = False,
    ) -> List[str] | List[Tuple[str, float]]:
        """
        Identify the ``topn`` most probable languages identified in ``text``.

        Args:
            text
            topn
            with_probs

        Returns:
            ISO 639-1 standard language code and corresponding prediction probability
            of the ``topn`` most probable languages.
        """
        if not self._is_valid_text(text):
            results = [("un", 1.0)]
        else:
            text_ = utils.to_collection(text, str, list)
            results = models.get_topn_preds_and_probs(
                self.model.predict(text_), topn, self.classes
            )[0]
        return [lang for lang, _ in results] if with_probs is False else results

    def _is_valid_text(self, text: str) -> bool:
        return any(char.isalpha() for char in text)


lang_identifier = LangIdentifier(
    models.LangIdentifierModelV2(),
    constants.DEFAULT_DATA_DIR.joinpath("lang_identifier"),
    version="2.0",
)
identify_lang = lang_identifier.identify_lang
