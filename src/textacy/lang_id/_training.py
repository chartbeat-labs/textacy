from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import sklearn.preprocessing
import thinc
from tqdm import tqdm
from thinc.api import Model

from . import models


def train_model(
    model: Model,
    *,
    train: Sequence[Tuple[str, str]],
    test: Sequence[Tuple[str, str]],
    n_iter: int,
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
    # binarize language labels
    # NOTE: thinc seems to require type "float32" arrays for training labels
    # errors otherwise... :/
    lb = sklearn.preprocessing.LabelBinarizer()
    lb.fit([lang for _, lang in train])
    # THIS NEXT LINE IS CRITICAL: we need to save the training class labels
    # but don't want to keep this label binarizer around; so, add it to the model
    model.layers[-1].attrs["classes"] = lb.classes_

    Y_train = lb.transform([lang for _, lang in train]).astype("float32")
    Y_test = lb.transform([lang for _, lang in test])

    # make sure data is on the right device?
    # Y_train = self.model.ops.asarray(Y_train)
    # Y_test = self.model.ops.asarray(Y_test)

    X_train = [text for text, _ in train]
    X_test = [text for text, _ in test]

    losser = thinc.api.CategoricalCrossentropy(normalize=True)
    optimizer = thinc.api.Adam(learn_rate)

    model.initialize(X=X_train[:10], Y=Y_train[:10])
    print(f"{'epoch':>5}  {'loss':>8}  {'score':>8}")
    # iterate over epochs
    for n in range(n_iter):
        loss = 0.0
        # iterate over batches
        batches = model.ops.multibatch(batch_size, X_train, Y_train, shuffle=True)
        for X, Y in tqdm(batches, leave=False):
            Yh, backprop = model.begin_update(X)
            dYh, loss_batch = losser(Yh, Y)
            loss += loss_batch
            backprop(dYh)
            model.finish_update(optimizer)
            optimizer.step_schedules()

        if optimizer.averages:
            with model.use_params(optimizer.averages):
                score = evaluate_model(model, X_test=X_test, Y_test=Y_test, batch_size=1000)
        else:
            score = evaluate_model(model, X_test=X_test, Y_test=Y_test, batch_size=1000)
        print(f"{n:>5}  {loss:>8.3f}  {score:>8.3f}")

    if optimizer.averages:
        with model.use_params(optimizer.averages):
            pred_langs = models.get_model_preds(
                model, X_test, model.layers[-1].attrs["classes"]
            )
    else:
        pred_langs = models.get_model_preds(
            model, X_test, model.layers[-1].attrs["classes"]
        )
    true_langs = list(lb.inverse_transform(Y_test))
    print(sklearn.metrics.classification_report(true_langs, pred_langs))
    return model


def evaluate_model(
    model,
    *,
    X_test: Sequence[str],
    Y_test : thinc.types.Array2d,
    batch_size: int,
) -> float:
    """
    Args:
        model
        X_test
        Y_test
        batch_size
    """
    correct = 0
    total = 0
    for X, Y in model.ops.multibatch(batch_size, X_test, Y_test):
        Yh = model.predict(X)
        for yh, y in zip(Yh, Y):
            correct += (y.argmax(axis=0) == yh.argmax(axis=0)).sum()
        total += len(Y)
    return float(correct / total)


def decaying_cyclic_triangular(
    decaying: thinc.types.Generator,
    cyclic_triangular: thinc.types.Generator,
    *,
    min_lr: float = 0.0,
) -> Iterable[float]:
    """
    Custom "schedule" that multiplicatively combines two built-in thinc schedules:
    :func:`thinc.api.decaying()` and :func:`thinc.api.cyclic_triangular()`

    Args:
        decaying
        cyclic_triangular
        min_lr: Minimum learning rate, after combination. Combined values that are
            smaller than this will be set equal to it.
    """
    for lr_decay, lr_ct in zip(decaying, cyclic_triangular):
        yield max(lr_decay * lr_ct, min_lr)
