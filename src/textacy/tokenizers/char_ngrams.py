from __future__ import annotations

from typing import Callable, Collection, Iterable, List, Optional, Tuple, Union

from spacy.tokens import Doc, Span

from .. import errors, types, utils


class CharNgramsTokenizer:
    """
    Transform each text in a sequence of texts into a sequence of character ngram tokens
    as strings, where ngrams for multiple "n"s are concatenated together for each text.

    Args:
        ns: Number of contiguous characters to include in each token.
            If an int, only ``ns`` n-grams are included as tokens; if a Collection[int],
            then n-grams for each n in ``ns`` are concatenated together per text.
        pad: If True, pad texts by adding ``n - 1`` underscore characters to each side;
            if False, leave text as-is.
        normalize: Optional callable or string alias for a callable that processes
            each text before transforming it into a sequence of character ngrams.
            If None, text is left as-is; if "lower", text is lowercased; otherwise,
            the callable is applied directly to each text.
    """

    def __init__(
        self,
        ns: int | Collection[int],
        *,
        pad: bool = False,
        normalize: Optional[str | Callable[[str], str]] = None,
    ):
        self.ns: tuple[int, ...] = utils.to_tuple(ns)
        self.pad = pad
        self.normalize = self._init_normalize(normalize)

    def __str__(self) -> str:
        return (
            "CharNgramsTokenizer("
            f"ns={self.ns}, pad={self.pad}, normalize={self.normalize}"
            ")"
        )

    def _init_normalize(
        self, normalize: Optional[str | Callable[[str], str]]
    ) -> Optional[Callable[[str], str]]:
        if normalize is None:
            return None
        elif callable(normalize):
            return normalize
        elif normalize == "lower":
            return lambda text: text.lower()
        else:
            raise ValueError()

    def fit(self, textables: Iterable[str | types.DocLike]) -> "CharNgramsTokenizer":
        return self

    def transform(
        self,
        textables: Iterable[str | types.DocLike],
    ) -> Iterable[Tuple[str, ...]]:
        """
        Convert a sequence of texts or textable spaCy objects (Doc or Span)
        into an ordered, nested sequence of character ngrams.

        Args:
            textables

        Yields:
            Ordered sequence of character ngrams as strings for next textable.
        """
        texts = (self._to_text(textable) for textable in textables)
        if self.normalize is not None:
            texts = (self.normalize(text) for text in texts)
        for text in texts:
            tokens: List[str] = []
            for n in self.ns:
                if self.pad is True and n > 1:
                    pad_chars = "_" * (n - 1)
                    text = f"{pad_chars}{text}{pad_chars}"
                tokens.extend(text[i : i + n] for i in range(len(text) - n + 1))
            yield tuple(tokens)

    def _to_text(self, textable: str | types.DocLike) -> str:
        if isinstance(textable, str):
            return textable
        elif isinstance(textable, (Doc, Span)):
            return textable.text
        else:
            raise TypeError(
                errors.type_invalid_msg("textable", textable, Union[str, types.DocLike])
            )
