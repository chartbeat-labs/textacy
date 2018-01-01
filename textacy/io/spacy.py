"""
Todo:
    Figure out a better / more efficient way to handle reading/writing of
    spacy docs. This situation is tolerable and currently unavoidable, but
    it's not *good*.
"""
from __future__ import absolute_import, print_function, unicode_literals

from spacy.tokens.doc import Doc as SpacyDoc

from .. import compat
from .utils import open_sesame


def read_spacy_docs(fname):
    """
    Read the contents of a pickle file at ``fname``, all at once but then
    streaming docs one at a time.

    Args:
        fname (str): Path to file on disk from which data will be read.

    Yields:
        ``spacy.Doc``: Next deserialized document.

    Note:
        The docs are pickled together, as a list, so they are all loaded into
        memory when reading from disk. Mind your RAM usage!
    """
    with open_sesame(fname, mode='rb') as f:
        for spacy_doc in compat.pickle.load(f):
            yield spacy_doc


def write_spacy_docs(data, fname, make_dirs=False):
    """
    Write one or more ``spacy.Doc`` s to disk at ``fname``, using pickle.

    Args:
        data (``spacy.Doc`` or Iterable[``spacy.Doc``]): A single ``spacy.Doc``
            or a sequence of ``spacy.Doc`` s to write to disk.
        fname (str): Path to file on disk to which data will be written.
        make_dirs (bool): If True, automatically create (sub)directories if
            not already present in order to write ``fname``.

    Note:
        The docs are pickled together, as a list, so they are all loaded
        into memory when writing to disk. Mind your RAM usage!
    """
    if isinstance(data, SpacyDoc):
        data = [data]
    with open_sesame(fname, mode='wb', make_dirs=make_dirs) as f:
        compat.pickle.dump(list(data), f, protocol=-1)
