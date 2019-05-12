"""
Matrix
------

Functions for reading from and writing to disk CSC and CSR sparse matrices
in numpy binary format.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import scipy.sparse as sp

from .utils import open_sesame


def read_sparse_matrix(filepath, kind="csc"):
    """
    Read the data, indices, indptr, and shape arrays from a ``.npz`` file on disk
    at ``filepath``, and return an instantiated sparse matrix.

    Args:
        filepath (str): Path to file on disk from which data will be read.
        kind ({'csc', 'csr'}): Kind of sparse matrix to instantiate.

    Returns:
        :class:`scipy.sparse.csc_matrix` or :class:`scipy.sparse.csr_matrix`:
        An instantiated sparse matrix, depending on the value of ``kind``.

    See Also:
        https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.io.html#numpy-binary-files-npy-npz
    """
    npz_file = np.load(filepath)
    if kind == "csc":
        return sp.csc_matrix(
            (npz_file["data"], npz_file["indices"], npz_file["indptr"]),
            shape=npz_file["shape"],
        )
    elif kind == "csr":
        return sp.csr_matrix(
            (npz_file["data"], npz_file["indices"], npz_file["indptr"]),
            shape=npz_file["shape"],
        )
    else:
        raise ValueError(
            'kind="{}" is invalid; valid values are {}'.format(kind, ["csc", "csr"])
        )


def write_sparse_matrix(data, filepath, compressed=True, make_dirs=False):
    """
    Write sparse matrix ``data`` to disk at ``filepath``, optionally compressed,
    into a single ``.npz`` file.

    Args:
        data (:class:`scipy.sparse.csc_matrix` or :class:`scipy.sparse.csr_matrix`)
        filepath (str): Path to file on disk to which data will be written.
            If ``filepath`` does not end in ``.npz``, that extension is
            automatically appended to the name.
        compressed (bool): If True, save arrays into a single file in compressed
            numpy binary format.
        make_dirs (bool): If True, automatically create (sub)directories if
            not already present in order to write ``filepath``.

    See Also:
        https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.io.html#numpy-binary-files-npy-npz
    """
    if not isinstance(data, (sp.csc_matrix, sp.csr_matrix)):
        raise TypeError(
            "`data` must be a scipy sparse csr or csc matrix, "
            'not "{}"'.format(type(data))
        )
    if make_dirs is True:
        _make_dirs(filepath, "w")
    if compressed is True:
        np.savez_compressed(
            filepath,
            data=data.data,
            indices=data.indices,
            indptr=data.indptr,
            shape=data.shape,
        )
    else:
        np.savez(
            filepath,
            data=data.data,
            indices=data.indices,
            indptr=data.indptr,
            shape=data.shape,
        )
