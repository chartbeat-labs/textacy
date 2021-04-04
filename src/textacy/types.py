from pathlib import Path
from typing import Callable, Iterable, Tuple, Union

from spacy.language import Language
from spacy.tokens import Doc, Span


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

Record = Tuple[str, dict]
DocData = Union[str, Record, Doc]

CorpusData = Union[str, Doc, Record, Iterable[str], Iterable[Doc], Iterable[Record]]

PathLike = Union[str, Path]
