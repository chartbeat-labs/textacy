"""
:mod:`textacy.types`: Definitions for common object types used throughout the package.
"""
import collections
from pathlib import Path
from typing import Callable, Iterable, Union

from spacy.language import Language
from spacy.tokens import Doc, Span, Token


# Record = Tuple[str, dict]  => let's use a namedtuple instead
Record = collections.namedtuple("Record", ["text", "meta"])
DocData = Union[str, Record, Doc]
CorpusData = Union[str, Record, Doc, Iterable[str], Iterable[Record], Iterable[Doc]]

LangLike = Union[str, Path, Language]
LangLikeInContext = Union[
    str,
    Path,
    Language,
    Callable[[str], str],
    Callable[[str], Path],
    Callable[[str], Language],
]

DocLike = Union[Doc, Span]
SpanLike = Union[Span, Token]

PathLike = Union[str, Path]

DocLikeToSpans = Callable[[DocLike], Iterable[Span]]
