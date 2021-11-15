"""
:mod:`textacy.types`: Definitions for common object types used throughout the package.
"""
from pathlib import Path
from typing import Any, Callable, Iterable, List, NamedTuple, Protocol, TypeVar, Union

from spacy.language import Language
from spacy.tokens import Doc, Span, Token


AnyVal = TypeVar("AnyVal")


# typed equivalent to Record = collections.namedtuple("Record", ["text", "meta"])
class Record(NamedTuple):
    text: str
    meta: dict


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


# typed equivalent to AugTok = collections.namedtuple("AugTok", ["text", "ws", "pos", "is_word", "syns"])
class AugTok(NamedTuple):
    """Minimal token data required for data augmentation transforms."""

    text: str
    ws: str
    pos: str
    is_word: bool
    syns: List[str]


class AugTransform(Protocol):
    def __call__(self, aug_toks: List[AugTok], **kwargs: Any) -> List[AugTok]:
        ...
