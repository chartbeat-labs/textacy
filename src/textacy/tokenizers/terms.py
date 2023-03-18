from __future__ import annotations

import operator
from functools import partial
from typing import Callable, Collection, Iterable, Optional

from cytoolz import itertoolz
from spacy.tokens import Span

from .. import extract, types


DocLikeToSpans = Callable[[types.DocLike], Iterable[Span]]


class TermsTokenizer:
    """
    Transform each ``Doc`` or ``Span`` in a sequence thereof into a sequence of terms
    as strings, where "terms" are the concatenated combination of constituent
    n-grams, entities, and/or noun chunks.

    Args:
        ngrams
        entities
        noun_chunks
        normalize
        dedupe
    """

    def __init__(
        self,
        *,
        ngrams: Optional[int | Collection[int] | DocLikeToSpans] = None,
        entities: Optional[bool | DocLikeToSpans] = None,
        noun_chunks: Optional[bool | DocLikeToSpans] = None,
        normalize: Optional[str | Callable[[Span], str]] = None,
        dedupe: bool = True,
    ):
        self.tokenizers = self._init_tokenizers(ngrams, entities, noun_chunks)
        self.normalize = self._init_normalize(normalize)
        self.dedupe = dedupe

    def __str__(self) -> str:
        return (
            "TermsTokenizer("
            f"tokenizers={self.tokenizers}, normalize={self.normalize}, dedupe={self.dedupe}"
            ")"
        )

    def _init_tokenizers(
        self, ngrams, entities, noun_chunks
    ) -> tuple[DocLikeToSpans, ...]:
        ngs_tokenizer = self._init_ngrams_tokenizer(ngrams)
        ents_tokenizer = self._init_entities_tokenizer(entities)
        ncs_tokenizer = self._init_noun_chunks_tokenizer(noun_chunks)
        tokenizers = tuple(
            tokenizer
            for tokenizer in [ngs_tokenizer, ents_tokenizer, ncs_tokenizer]
            if tokenizer is not None
        )
        if not tokenizers:
            raise ValueError("at least one term tokenizer must be specified")
        else:
            return tokenizers

    def _init_ngrams_tokenizer(
        self, ngrams: Optional[int | Collection[int] | DocLikeToSpans]
    ) -> Optional[Callable[[types.DocLike], Iterable[Span]]]:
        if ngrams is None:
            return None
        elif callable(ngrams):
            return ngrams
        elif isinstance(ngrams, int):
            return partial(extract.ngrams, n=ngrams)
        elif isinstance(ngrams, Collection) and all(
            isinstance(ng, int) for ng in ngrams
        ):
            return partial(_concat_extract_ngrams, ns=ngrams)
        else:
            raise TypeError()

    def _init_entities_tokenizer(
        self, entities: Optional[bool | DocLikeToSpans]
    ) -> Optional[DocLikeToSpans]:
        if entities is None:
            return None
        elif callable(entities):
            return entities
        elif isinstance(entities, bool):
            return extract.entities
        else:
            raise TypeError()

    def _init_noun_chunks_tokenizer(
        self,
        noun_chunks: Optional[bool | DocLikeToSpans],
    ) -> Optional[DocLikeToSpans]:
        if noun_chunks is None:
            return None
        elif callable(noun_chunks):
            return noun_chunks
        elif isinstance(noun_chunks, bool):
            return extract.noun_chunks
        else:
            raise TypeError()

    def _init_normalize(
        self, normalize: Optional[str | Callable[[Span], str]]
    ) -> Callable[[Span], str]:
        if not normalize:
            return operator.attrgetter("text")
        elif normalize == "lower":
            return lambda span: span.text.lower()
        elif normalize == "lemma":
            return lambda span: span.lemma_
        elif callable(normalize):
            return normalize
        else:
            raise ValueError()

    def fit(self, doclikes: Iterable[types.DocLike]) -> "TermsTokenizer":
        return self

    def transform(self, doclikes: Iterable[types.DocLike]) -> Iterable[tuple[str, ...]]:
        """
        Convert a sequence of spaCy Docs or Spans into an ordered, nested sequence
        of terms as strings.

        Args:
            doclikes

        Yields:
            Ordered sequence of terms as strings for next Doc or Span.
        """
        normalize_ = self.normalize
        for doclike in doclikes:
            terms = itertoolz.concat(
                tokenizer(doclike) for tokenizer in self.tokenizers
            )
            if self.dedupe is True:
                terms = itertoolz.unique(terms, lambda span: (span.start, span.end))
            yield tuple(normalize_(term) for term in terms)


def _concat_extract_ngrams(
    doclike: types.DocLike, ns: Collection[int]
) -> Iterable[Span]:
    for n in ns:
        ngrams = extract.ngrams(doclike, n=n)
        for ngram in ngrams:
            yield ngram


# TODO: do we want to try lazy imports here?
# import importlib
# def __getattr__(name: str) -> Any:
#     if name == "extract":
#         return importlib.import_module(f".{name}", package="textacy")
