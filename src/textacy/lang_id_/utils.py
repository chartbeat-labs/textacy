from __future__ import annotations

from typing import List, Sequence, Tuple

import thinc
from thinc.api import Model


def train_model(
    model: Model,
    *,
    train: Tuple[Sequence[str], thinc.types.Array2d],
    test: Tuple[Sequence[str], thinc.types.Array2d],
    n_iter: int = 3,
    batch_size: int | thinc.types.Generator = 32,
    learn_rate: float | List[float] | thinc.types.Generator = 0.001,
) -> Model:
    """
    Args:
        model
        ...
        n_iter
        batch_size
        learn_rate
    """
    from tqdm import tqdm  # HACK: hiding this non-required dep inside a function

    # make sure data is on the right device
    X_train = model.ops.asarray(train[0])
    Y_train = model.ops.asarray(train[1])
    X_test = model.ops.asarray(test[0])
    Y_test = model.ops.asarray(test[1])

    losser = thinc.api.CategoricalCrossentropy(normalize=True)
    optimizer = thinc.api.Adam(learn_rate)

    model.initialize(X=X_train[:100], Y=Y_train[:100])
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

        score = evaluate_model(model, X_test=X_test, Y_test=Y_test, batch_size=128)
        print(f"{n:>5}  {loss:>8.3f}  {score:>8.3f}")

    return model


def evaluate_model(
    model: Model,
    *,
    X_test: thinc.types.Array1d,
    Y_test : thinc.types.Array2d,
    batch_size: int,
) -> float:
    correct = 0
    total = 0
    for X, Y in model.ops.multibatch(batch_size, X_test, Y_test):
        Yh = model.predict(X)
        for yh, y in zip(Yh, Y):
            correct += (y.argmax(axis=0) == yh.argmax(axis=0)).sum()
        total += len(Y)
    return float(correct / total)
