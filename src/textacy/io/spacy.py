"""
:mod:`textacy.io.spacy`: Functions for reading from and writing to disk spacy documents
in either pickle or binary format. Be warned: Both formats have pros and cons.
"""
from __future__ import annotations

import pickle
from typing import Iterable, Literal, Optional

from spacy.tokens import Doc, DocBin

from .. import errors, spacier, types
from . import utils as io_utils


FormatType = Literal["binary", "pickle"]


def read_spacy_docs(
    filepath: types.PathLike,
    *,
    format: FormatType = "binary",
    lang: Optional[types.LangLike] = None,
) -> Iterable[Doc]:
    """
    Read the contents of a file at ``filepath``, written in binary or pickle format.

    Args:
        filepath: Path to file on disk from which data will be read.
        format: Format of the data that was written to disk.
            If "binary", uses :class:`spacy.tokens.DocBin` to deserialize data;
            if "pickle", uses python's stdlib ``pickle``.

            .. warning:: Docs written in pickle format were saved all together
               as a list, which means they're all loaded into memory at once
               before streaming one by one. Mind your RAM usage, especially when
               reading many docs!

        lang: Language with which spaCy originally processed docs, represented as
            the full name of or path on disk to the pipeline, or an already instantiated
            pipeline instance.
            Note that this is only required when ``format`` is "binary".

    Yields:
        Next deserialized document.

    Raises:
        ValueError: if format is not "binary" or "pickle", or if ``lang`` is None
            when ``format="binary"``
    """
    if format == "binary":
        if lang is None:
            raise ValueError(
                "lang=None is invalid. When format='binary', a `spacy.Language` "
                "(well, its associated `spacy.Vocab`) is required to deserialize "
                "the binary data. Note that this should be the same language pipeline "
                "used when processing the original docs!"
            )
        else:
            lang = spacier.utils.resolve_langlike(lang)
        docbin = DocBin().from_disk(filepath)
        for doc in docbin.get_docs(lang.vocab):
            yield doc

    elif format == "pickle":
        with io_utils.open_sesame(filepath, mode="rb") as f:
            for spacy_doc in pickle.load(f):
                yield spacy_doc

    else:
        raise ValueError(
            errors.value_invalid_msg("format", format, {"binary", "pickle"})
        )


def write_spacy_docs(
    data: Doc | Iterable[Doc],
    filepath: types.PathLike,
    *,
    make_dirs: bool = False,
    format: FormatType = "binary",
    attrs: Optional[Iterable[str]] = None,
    store_user_data: bool = False,
) -> None:
    """
    Write one or more ``Doc`` s to disk at ``filepath`` in binary or pickle format.

    Args:
        data: A single ``Doc`` or a sequence of ``Doc`` s to write to disk.
        filepath: Path to file on disk to which data will be written.
        make_dirs: If True, automatically create (sub)directories
            if not already present in order to write ``filepath``.
        format: Format of the data written to disk.
            If "binary", uses :class:`spacy.tokens.DocBin` to serialie data;
            if "pickle", uses python's stdlib ``pickle``.

            .. warning:: When writing docs in pickle format, all the docs in ``data``
               must be saved as a list, which means they're all loaded into memory.
               Mind your RAM usage, especially when writing many docs!

        attrs: List of attributes to serialize if ``format`` is "binary". If None,
            spaCy's default values are used; see here: https://spacy.io/api/docbin#init
        store_user_data: If True, write :attr`Doc.user_data` and the values of custom
            extension attributes to disk; otherwise, don't.

    Raises:
        ValueError: if format is not "binary" or "pickle"
    """
    if isinstance(data, Doc):
        data = [data]
    if format == "binary":
        kwargs = {"docs": data, "store_user_data": store_user_data}
        if attrs is not None:
            kwargs["attrs"] = list(attrs)
        docbin = DocBin(**kwargs)
        docbin.to_disk(filepath)
    elif format == "pickle":
        if store_user_data is False:
            data = _clear_docs_user_data(data)
        with io_utils.open_sesame(filepath, mode="wb", make_dirs=make_dirs) as f:
            pickle.dump(list(data), f, protocol=-1)
    else:
        raise ValueError(
            errors.value_invalid_msg("format", format, {"binary", "pickle"})
        )


def _clear_docs_user_data(docs: Iterable[Doc]) -> Iterable[Doc]:
    # TODO: figure out if/how to clear out custom doc extension values
    for doc in docs:
        doc.user_data.clear()
        yield doc
