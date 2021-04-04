"""
:mod:`textacy.io.matrix`: Functions for reading from and writing to disk CSC and CSR
sparse matrices in numpy binary format.
"""
from __future__ import annotations

from typing import Union

import numpy as np
import scipy.sparse as sp

from .. import errors, types, utils
from . import utils as io_utils


def read_sparse_matrix(
    filepath: types.PathLike, *, kind: str = "csc",
) -> sp.csc_matrix | sp.csr_matrix:
    """
    Read the data, indices, indptr, and shape arrays from a ``.npz`` file on disk
    at ``filepath``, and return an instantiated sparse matrix.

    Args:
        filepath: Path to file on disk from which data will be read.
        kind ({'csc', 'csr'}): Kind of sparse matrix to instantiate.

    Returns:
        An instantiated sparse matrix, whose type depends on the value of ``kind``.

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
        raise ValueError(errors.value_invalid_msg("kind", kind, {"csc", "csr"}))


def write_sparse_matrix(
    data: sp.csc_matrix | sp.csr_matrix,
    filepath: types.PathLike,
    *,
    compressed: bool = True,
    make_dirs: bool = False,
) -> None:
    """
    Write sparse matrix ``data`` to disk at ``filepath``, optionally compressed,
    into a single ``.npz`` file.

    Args:
        data
        filepath: Path to file on disk to which data will be written. If ``filepath``
            does not end in ``.npz``, that extension is automatically appended to the name.
        compressed: If True, save arrays into a single file in compressed numpy binary format.
        make_dirs: If True, automatically create (sub)directories
            if not already present in order to write ``filepath``.

    See Also:
        https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.io.html#numpy-binary-files-npy-npz
    """
    if not isinstance(data, (sp.csc_matrix, sp.csr_matrix)):
        raise TypeError(
            errors.type_invalid_msg(
                "data", type(data), Union[sp.csc_matrix, sp.csr_matrix]
            )
        )
    filepath = utils.to_path(filepath).resolve()
    if make_dirs is True:
        io_utils._make_dirs(filepath, "w")
    if compressed is True:
        np.savez_compressed(
            str(filepath),
            data=data.data,
            indices=data.indices,
            indptr=data.indptr,
            shape=data.shape,
        )
    else:
        np.savez(
            str(filepath),
            data=data.data,
            indices=data.indices,
            indptr=data.indptr,
            shape=data.shape,
        )
