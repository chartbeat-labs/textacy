"""
:mod:`textacy.types`: Definitions for common object types used throughout the package.
"""
from pathlib import Path
from typing import Any, Callable, Iterable, NamedTuple, Protocol, TypeVar, Union

from spacy.language import Language
from spacy.tokens import Doc, Span, Token


AnyVal = TypeVar("AnyVal")
AnyStr = Union[str, bytes]

PathLike = Union[str, Path]

DocLike = Union[Doc, Span]
SpanLike = Union[Span, Token]
DocLikeToSpans = Callable[[DocLike], Iterable[Span]]
DocOrTokens = Union[Doc, Iterable[Token]]

LangLike = Union[str, Path, Language]
LangLikeInContext = Union[
    PathLike,
    Language,
    Callable[[str], str],
    Callable[[str], Path],
    Callable[[str], Language],
]


# typed equivalent to Record = collections.namedtuple("Record", ["text", "meta"])
class Record(NamedTuple):
    text: str
    meta: dict


DocData = Union[str, Record, Doc]
CorpusData = Union[str, Record, Doc, Iterable[str], Iterable[Record], Iterable[Doc]]


# typed equivalent to AugTok = collections.namedtuple("AugTok", ["text", "ws", "pos", "is_word", "syns"])
class AugTok(NamedTuple):
    """Minimal token data required for data augmentation transforms."""

    text: str
    ws: str
    pos: str
    is_word: bool
    syns: list[str]


class AugTransform(Protocol):
    def __call__(self, aug_toks: list[AugTok], **kwargs: Any) -> list[AugTok]:
        ...


class DocExtFunc(Protocol):
    def __call__(self, doc: Doc, *args: Any, **kwargs: Any) -> Any:
        ...
