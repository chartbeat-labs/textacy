from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import thinc
import thinc.layers
import thinc.types
from cytoolz import itertoolz
from thinc.api import Model, chain, concatenate


def get_model_preds(model: Model, texts: List[str], classes: np.ndarray) -> List[str]:
    """
    Get model predictions for multiple texts as class labels rather than as a 2dim
    matrix of prediction probabilities.
    """
    # predict in batches, otherwise memory blows UP
    results = (
        result
        for texts_pt in itertoolz.partition_all(1000, texts)
        for result in get_topn_preds_and_probs(model.predict(texts_pt), 1, classes)
    )
    return [lang for result in results for lang, _ in result]


def get_topn_preds_and_probs(
    preds: np.ndarray,
    topn: int,
    classes: np.ndarray,
) -> List[List[Tuple[str, float]]]:
    # TODO
    # if only need 1 (max) value, use faster numpy ops?
    # if topn == 1:
    #     idxs = np.argmax(preds, axis=1)
    #     pred_probs = np.max(preds, axis=1)
    #     pred_langs = self.classes[idxs]
    #     return list(zip(pred_langs, pred_probs))
    # otherwise, do the full array sorts to get topn max
    # else:
    idxs = np.argsort(preds, axis=1)[:, ::-1][:, :topn]
    pred_probs = np.sort(preds, axis=1)[:, ::-1][:, :topn]
    pred_langs = classes[idxs]
    return [list(zip(pred_langs[i], pred_probs[i])) for i in range(pred_probs.shape[0])]


def LangIdentifierModelV2(
    ns: Sequence[int] = (1, 2, 3),
    embed_dim: int = 100,
    hidden_width: int = 512,
    dropout: Optional[float] = 0.1,
) -> Model[List[str], thinc.types.Floats2d]:
    """
    Build a language identification model inspired by Google's CLD3.

    Args:
        ns: Set of "n" for which character "n"-grams are extracted from input texts.
            If 1, only unigrams (single characters) are used; if [1, 2], then both
            unigrams and bigrams are used; and so on.
        embed_dim: Size of the vectors into which each set of ngrams are embedded.
        hidden_width: Width of the dense layer with Relu activation, just before
            the final prediction (Softmax) layer.
        dropout: Dropout rate to avoid overfitting.

    Returns:
        Thinc :class:`Model`.
    """
    with Model.define_operators({">>": chain}):
        model = (
            MultiCharNgramsEmbedding(
                ns=list(ns),
                max_chars=1000,
                lower=True,
                num_vectors=[2000 * n for n in ns],
                embed_dims=embed_dim,
                dropout=dropout,
            )
            >> thinc.layers.Relu(
                nI=embed_dim * len(ns),
                nO=hidden_width,
                dropout=dropout,
            )
            >> thinc.layers.Softmax(nI=hidden_width)
        )
    return model


def MultiCharNgramsEmbedding(
    ns: List[int],
    max_chars: int,
    lower: bool,
    num_vectors: int | List[int],
    embed_dims: int | List[int],
    dropout: Optional[float],
) -> Model[List[str], thinc.types.Floats1d]:
    """
    Args:
        ns
        max_chars
        lower
        num_vectors
        embed_dims
        dropout
    """
    numn = len(ns)
    num_vectors = [num_vectors] * numn if isinstance(num_vectors, int) else num_vectors
    embed_dims = [embed_dims] * numn if isinstance(embed_dims, int) else embed_dims
    with Model.define_operators({">>": chain}):
        model = concatenate(
            *[
                CharNgramsEmbedding(
                    n=n,
                    max_chars=max_chars,
                    lower=lower,
                    num_vectors=nvec,
                    embed_dim=edim,
                    dropout=dropout,
                )
                for n, nvec, edim in zip(ns, num_vectors, embed_dims)
            ]
        )
    return model


def CharNgramsEmbedding(
    n: int,
    max_chars: int,
    lower: bool,
    num_vectors: int,
    embed_dim: int,
    dropout: Optional[float],
) -> Model[List[str], thinc.types.Floats1d]:
    """
    Args:
        n
        max_chars
        lower
        num_vectors
        embed_dim
        dropout
    """
    with Model.define_operators({">>": chain}):
        model = (
            text_to_char_ngrams(n, max_chars, lower)
            >> thinc.layers.strings2arrays()
            >> thinc.layers.with_array(
                thinc.layers.HashEmbed(
                    nO=embed_dim,
                    nV=num_vectors,
                    dropout=dropout,
                    column=0,
                )
            )
            >> thinc.layers.list2ragged()
            >> thinc.layers.reduce_mean()
        )
    return model


def text_to_char_ngrams(
    n: int,
    max_chars: int,
    lower: bool,
) -> Model[List[str], List[List[str]]]:
    """
    Custom data type transfer thinc layer that transforms a sequence of text strings
    into a sequence of sequence of character ngram strings. Like this::

        ["a short text.", "another text."] => [["a ", " s", "sh", "ho", ...], ...]

    Args:
        n: Number of adjacent characters to combine into an ngram.
        max_chars: Max number of characters from the start of the text to transform
            into character ngrams.
        lower: If True, lowercase text before extracting character ngrams; otherwise,
            leave text casing as-is.
    """

    def forward(
        model: Model, texts: List[str], is_train: bool
    ) -> Tuple[List[List[str]], Callable]:
        if lower is True:
            texts = [text[:max_chars].lower() for text in texts]
        else:
            texts = [text[:max_chars] for text in texts]
        if n == 1:
            char_ngs = [list(text) for text in texts]
        else:
            char_ngs = [
                [text[i : i + n] for i in range(len(text) - n + 1)] for text in texts
            ]

        def backprop(dY):
            return []

        return (char_ngs, backprop)

    return Model(
        "texts_to_char_ngrams",
        forward,
        attrs={"n": n, "max_chars": max_chars, "lower": lower},
    )
