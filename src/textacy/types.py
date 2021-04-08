"""
:mod:`textacy.types`: Definitions for common object types used throughout the package.
"""
from pathlib import Path
from typing import Callable, Iterable, Tuple, Union

from spacy.language import Language
from spacy.tokens import Doc, Span, Token


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
TokLike = Union[Token, Span]

Record = Tuple[str, dict]
DocData = Union[str, Record, Doc]
CorpusData = Union[str, Record, Doc, Iterable[str], Iterable[Record], Iterable[Doc]]

PathLike = Union[str, Path]

DocLikeToSpans = Callable[[DocLike], Iterable[Span]]
