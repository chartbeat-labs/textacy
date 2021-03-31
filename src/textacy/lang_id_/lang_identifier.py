from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

import numpy as np
import thinc
from thinc.api import Model

from . import models
from .. import constants, utils


LOGGER = logging.getLogger()


class LangIdentifier:
    """
    Args:
        model
        fpath
    """

    def __init__(
        self,
        model: Model,
        data_dir=constants.DEFAULT_DATA_DIR.joinpath("lang_identifier")
    ):
        self.model = model
        # TODO: should we just pass the full path?
        self.model_fpath = utils.to_path(data_dir).joinpath("lang-identifier-v2.bin")
        self.min_prob_reliable = 0.5
        self._classes = None

    @property
    def classes(self):
        if self._classes is None:
            self._classes = self.model.layers[-1].attrs["classes"]
        return self._classes

    def save_model(self):
        LOGGER.info("saving LangIdentifier model to %s", self.model_fpath)
        self.model.to_disk(self.model_fpath)

    def load_model(self):
        LOGGER.debug("loading LangIdentifier model from %s", self.model_fpath)
        self.model.from_disk(self.model_fpath)

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
        # return [lang for lang, _ in results]
        return [lang for result in results for lang, _ in result]

    def _get_topn_preds_and_probs(
        self,
        preds: np.ndarray,
        topn: int,
    ) -> List[List[Tuple[str, float]]]:
        # TODO
        # if only need 1 (max) value, use faster numpy ops
        # if topn == 1:
        #     idxs = np.argmax(preds, axis=1)
        #     pred_probs = np.max(preds, axis=1)
        #     pred_langs = self.classes[idxs]
        #     return list(zip(pred_langs, pred_probs))
        # otherwise, do the full array sorts to get topn max
        # else:
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

    def train_model(
        self,
        *,
        train: Sequence[Tuple[str, str]],
        test: Sequence[Tuple[str, str]],
        n_iter: int = 3,
        batch_size: int | thinc.types.Generator = 32,
        learn_rate: float | List[float] | thinc.types.Generator = 0.001,
    ) -> Model:
        """
        Args:
            model
            train
            test
            n_iter
            batch_size
            learn_rate
        """
        # hiding training-only imports here
        import sklearn.preprocessing
        from tqdm import tqdm

        # binarize language labels
        # NOTE: thinc seems to require type "float32" arrays for training labels
        # errors otherwise... :/
        lb = sklearn.preprocessing.LabelBinarizer()
        lb.fit([lang for _, lang in train])
        # THIS NEXT LINE IS CRITICAL: we need to save the training class labels
        # but don't want to keep this label binarizer around; so, add it to the model
        self.model.layers[-1].attrs["classes"] = lb.classes_

        Y_train = lb.transform([lang for _, lang in train]).astype("float32")
        Y_test = lb.transform([lang for _, lang in test])

        # # make sure data is on the right device?
        # Y_train = self.model.ops.asarray(Y_train)
        # Y_test = self.model.ops.asarray(Y_test)

        X_train = [text for text, _ in train]
        X_test = [text for text, _ in test]

        losser = thinc.api.CategoricalCrossentropy(normalize=True)
        optimizer = thinc.api.Adam(learn_rate)

        self.model.initialize(X=X_train[:50], Y=Y_train[:50])
        print(f"{'epoch':>5}  {'loss':>8}  {'score':>8}")
        # iterate over epochs
        for n in range(n_iter):
            loss = 0.0
            # iterate over batches
            batches = self.model.ops.multibatch(batch_size, X_train, Y_train, shuffle=True)
            for X, Y in tqdm(batches, leave=False):
                Yh, backprop = self.model.begin_update(X)
                dYh, loss_batch = losser(Yh, Y)
                loss += loss_batch
                backprop(dYh)
                self.model.finish_update(optimizer)
                optimizer.step_schedules()

            if optimizer.averages:
                with self.model.use_params(optimizer.averages):
                    score = self.evaluate_model(
                        X_test=X_test, Y_test=Y_test, batch_size=1000
                    )
            else:
                score = self.evaluate_model(
                    X_test=X_test, Y_test=Y_test, batch_size=1000
                )
            print(f"{n:>5}  {loss:>8.3f}  {score:>8.3f}")

        true_langs = list(lb.inverse_transform(Y_test))
        pred_langs = self._get_model_preds(X_test)
        print(sklearn.metrics.classification_report(true_langs, pred_langs))

    def evaluate_model(
        self,
        *,
        X_test: Sequence[str],
        Y_test : thinc.types.Array2d,
        batch_size: int,
    ) -> float:
        correct = 0
        total = 0
        for X, Y in self.model.ops.multibatch(batch_size, X_test, Y_test):
            Yh = self.model.predict(X)
            for yh, y in zip(Yh, Y):
                correct += (y.argmax(axis=0) == yh.argmax(axis=0)).sum()
            total += len(Y)
        return float(correct / total)


lang_identifier = LangIdentifier(models.LangIdentifierModelV2(embed_dim=100))
identify_lang = lang_identifier.identify_lang
