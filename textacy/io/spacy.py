"""
spaCy
-----

Functions for reading from and writing to disk spacy documents in either
pickle or binary format. Be warned: Both formats have pros and cons.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from srsly import msgpack
from spacy.language import Language
from spacy.tokens import Doc

from .. import cache
from .. import compat
from .utils import open_sesame
from ..utils import deprecated


def read_spacy_docs(filepath, format="pickle", lang=None):
    """
    Read the contents of a file at ``filepath``, written either in pickle or binary
    format.

    Args:
        filepath (str): Path to file on disk from which data will be read.
        format ({"pickle", "binary"}): Format of the data that was written to disk.
            If 'pickle', use ``pickle`` in python's stdlib; if 'binary', use
            the 3rd-party ``msgpack`` library.

            .. warning:: Docs written in pickle format were saved all together
               as a list, which means they're all loaded into memory at once
               before streaming one by one. Mind your RAM usage, especially when
               reading many docs!

            .. warning:: When writing docs in binary format, spaCy's built-in
               ``spacy.Doc.to_bytes()`` method is used, but when reading the data
               back in :func:`read_spacy_docs()`, experimental and *unofficial*
               work-arounds are used to allow for all the docs in ``data`` to be
               read from the same file. If spaCy changes, this code could break,
               so use this functionality at your own risk!

        lang (str or ``spacy.Language``): Already-instantiated ``spacy.Language``
            object, or the string name by which it can be loaded, used to process
            the docs written to disk at ``filepath``. Note that this is only applicable
            when ``format="binary"``.

    Yields:
        :class:`spacy.tokens.Doc`: Next deserialized document.

    Raises:
        ValueError: if format is not "pickle" or "binary", or if ``lang`` is not
            provided when ``format="binary"``
    """
    if format == "pickle":
        with open_sesame(filepath, mode="rb") as f:
            for spacy_doc in compat.pickle.load(f):
                yield spacy_doc
    elif format == "binary":
        if lang is None:
            raise ValueError(
                "When format='binary', a `spacy.Language` (and its associated "
                "`spacy.Vocab`) is required to deserialize the binary data; "
                "and these should be the same as were used when processing "
                "the original docs!"
            )
        elif isinstance(lang, Language):
            vocab = lang.vocab
        elif isinstance(lang, compat.unicode_):
            vocab = cache.load_spacy_lang(lang).vocab
        else:
            raise ValueError(
                "lang = '{}' is invalid; must be a str or `spacy.Language`"
            )
        with open_sesame(filepath, mode="rb") as f:
            unpacker = msgpack.Unpacker(f, raw=False, unicode_errors="strict")
            for msg in unpacker:

                # NOTE: The following code has been adapted from spaCy's
                # built-in ``spacy.Doc.from_bytes()``. If that functionality
                # changes, the following will probably break...

                # Msgpack doesn't distinguish between lists and tuples, which is
                # vexing for user data. As a best guess, we *know* that within
                # keys, we must have tuples. In values we just have to hope
                # users don't mind getting a list instead of a tuple.
                if "user_data_keys" in msg:
                    user_data_keys = msgpack.loads(
                        msg["user_data_keys"], use_list=False
                    )
                    user_data_values = msgpack.loads(msg["user_data_values"])
                    user_data = {
                        key: value
                        for key, value in compat.zip_(user_data_keys, user_data_values)
                    }
                else:
                    user_data = None

                text = msg["text"]
                attrs = msg["array_body"]
                words = []
                spaces = []
                start = 0
                for i in compat.range_(attrs.shape[0]):
                    end = start + int(attrs[i, 0])
                    has_space = int(attrs[i, 1])
                    words.append(text[start: end])
                    spaces.append(bool(has_space))
                    start = end + has_space

                spacy_doc = Doc(
                    vocab, words=words, spaces=spaces, user_data=user_data
                )
                spacy_doc = spacy_doc.from_array(msg["array_head"][2:], attrs[:, 2:])
                if "sentiment" in msg:
                    spacy_doc.sentiment = msg["sentiment"]
                if "tensor" in msg:
                    spacy_doc.tensor = msg["tensor"]
                yield spacy_doc
    else:
        raise ValueError(
            "format = '{}' is invalid; value must be one of {}".format(
                format, {"pickle", "binary"}
            )
        )


def write_spacy_docs(
    data, filepath, make_dirs=False, format="pickle", exclude=("tensor",), include_tensor=None
):
    """
    Write one or more ``Doc`` s to disk at ``filepath`` in either pickle or
    binary format.

    Args:
        data (:class:`spacy.tokens.Doc` or Iterable[:class:`spacy.tokens.Doc`]):
            A single ``Doc`` or a sequence of ``Doc`` s to write to disk.
        filepath (str): Path to file on disk to which data will be written.
        make_dirs (bool): If True, automatically create (sub)directories if
            not already present in order to write ``filepath``.
        format ({"pickle", "binary"}): Format of the data written to disk.
            If "pickle", use python's stdlib ``pickle``; if "binary", use
            the 3rd-party ``msgpack`` library.

            .. warning:: When writing docs in pickle format, all the docs in ``data``
               must be saved as a list, which means they're all loaded into memory.
               Mind your RAM usage, especially when writing many docs!

            .. warning:: When writing docs in binary format, spaCy's built-in
               ``spacy.Doc.to_bytes()`` method is used, but when reading the data
               back in :func:`read_spacy_docs()`, experimental and *unofficial*
               work-arounds are used to allow for all the docs in ``data`` to be
               read from the same file. If spaCy changes, this code could break,
               so use this functionality at your own risk!

        exclude (List[str]): String names of serialization fields to exclude;
            see https://spacy.io/api/doc#serialization-fields for options.
            By default, excludes tensors in order to reproduce existing behavior
            of ``include_tensor=False``.
        include_tensor (bool): DEPRECATED! Use ``exclude`` instead.
            If False, ``Doc`` tensors are not written
            to disk; otherwise, they are. Note that this is only applicable when
            ``format="binary"``. Also note that including tensors *significantly*
            increases the file size of serialized docs.

    Raises:
        ValueError: if format is not "pickle" or "binary"
    """
    if include_tensor is not None:
        deprecated(
            "Use `exclude=('tensor',)` instead of `include_tensor=True`, since "
            "spacy has converged on this standard for usage.",
            action="once",
        )
        if include_tensor is False and "tensor" not in exclude:
            exclude = ["tensor"] + list(exclude)
        elif include_tensor is True and "tensor" in exclude:
            exclude = [field for field in exclude if field != "tensor"]

    if isinstance(data, Doc):
        data = [data]
    if format == "pickle":
        with open_sesame(filepath, mode="wb", make_dirs=make_dirs) as f:
            compat.pickle.dump(list(data), f, protocol=-1)
    elif format == "binary":
        with open_sesame(filepath, mode="wb", make_dirs=make_dirs) as f:
            for spacy_doc in data:
                f.write(spacy_doc.to_bytes(exclude=exclude))
    else:
        raise ValueError(
            "format = '{}' is invalid; value must be one of {}".format(
                format, {"pickle", "binary"}
            )
        )
