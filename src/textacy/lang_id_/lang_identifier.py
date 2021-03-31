from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
from thinc.api import Model

from . import model
from .. import constants, utils


LOGGER = logging.getLogger()


class LangIdentifier:
    """
    """

    def __init__(self, data_dir=constants.DEFAULT_DATA_DIR.joinpath("lang_identifier")):
        self.data_dir = utils.to_path(data_dir)
        self._model_fpath = self.data_dir.joinpath("lang-identifier-v2.bin")
        self._model = None
        self.min_prob_reliable = 0.5
        self._classes = None

    @property
    def model(self):
        if self._model is None:
            self._model = model.LangIdentifierModelV2()
        return self._model

    @property
    def classes(self):
        if self._classes is None:
            self._classes = self.model.layers[-1].attrs["classes"]
        return self._classes

    def save_model(self):
        LOGGER.info("saving LangIdentifier model to %s", self._model_fpath)
        self.model.to_disk(self._model_fpath)

    def load_model(self):
        LOGGER.debug("loading LangIdentifier model from %s", self._model_fpath)
        self.model.from_disk(self._model_fpath)

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
            text_ = utils.to_collection(text[: self.max_text_len], str, list)
            preds = self.model.predict(text_)
            result = self._get_topn_preds_and_probs(preds, 1)[0]
        if with_probs is False:
            return result[0]
        else:
            return result

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
            text_ = utils.to_collection(text[: self.max_text_len], str, list)
            preds = self.model.predict(text_)
            results = self._get_topn_preds_and_probs(preds, topn)[0]
        if with_probs is False:
            return [lang for lang, _ in results]
        else:
            return results

    def _get_model_preds(self, texts: List[str]) -> List[str]:
        preds = self.model.predict(texts)
        results = self._get_topn_preds_and_probs(preds, 1)
        return [lang for result in results for lang, _ in result]

    def _get_topn_preds_and_probs(
        self,
        preds: np.ndarray,
        topn: int,
    ) -> List[List[Tuple[str, float]]]:
        # TODO: decide if we should include a simpler path, something like this
        # if topn == 1:
        #     idxs = np.argmax(preds, axis=1)
        #     pred_probs = np.max(preds, axis=1)
        #     pred_langs = classes_[idxs]
        idxs = np.argsort(preds, axis=1)[:,::-1][:,:topn]
        pred_probs = np.sort(preds, axis=1)[:,::-1][:,:topn]
        pred_langs = self.classes[idxs]
        return [
            list(zip(pred_langs[i], pred_probs[i]))
            for i in range(pred_probs.shape[0])
        ]

    def _is_valid_text(self, text: str) -> bool:
        return any(char.isalpha() for char in text)

    def _is_reliable_pred(self, pred) -> bool:
        return pred >= self.min_prob_reliable


lang_identifier = LangIdentifier()
identify_lang = lang_identifier.identify_lang
