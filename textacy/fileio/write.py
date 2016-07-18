"""
Module with functions for writing content to disk in common formats.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import json

from numpy import savez, savez_compressed
from scipy.sparse import csc_matrix, csr_matrix
from spacy.tokens.doc import Doc as SpacyDoc

from textacy.fileio import open_sesame, make_dirs


def write_json(json_object, filepath, mode='wt', encoding=None):
    """
    Write JSON object all at once to disk at ``filepath``.

    Args:
        json_object (json): valid JSON object to be written
        filepath (str): /path/to/file on disk to which json object will be written,
            such as a JSON array; for example::

                [
                    {"title": "Harrison Bergeron", "text": "The year was 2081, and everybody was finally equal."},
                    {"title": "2BR02B", "text": "Everything was perfectly swell."}
                ]

        mode (str, optional)
        encoding (str, optional)
    """
    with open_sesame(filepath, mode=mode, encoding=encoding) as f:
        f.write(json.dumps(json_object, f, ensure_ascii=False))


def write_json_lines(json_objects, filepath, mode='wt', encoding=None):
    """
    Iterate over a stream of JSON objects, writing each to a separate line in
    file ``filepath`` but without a top-level JSON object (e.g. array).

    Args:
        json_objects (iterable(json)): sequence of valid JSON objects to be written
        filepath (str): /path/to/file on disk to which JSON objects will be written,
            where each line in the file is its own json object; for example::

                {"title": "Harrison Bergeron", "text": "The year was 2081, and everybody was finally equal."}\n
                {"title": "2BR02B", "text": "Everything was perfectly swell."}

        mode (str, optional)
        encoding (str, optional)
    """
    with open_sesame(filepath, mode=mode, encoding=encoding) as f:
        for json_object in json_objects:
            f.write(json.dumps(json_object, ensure_ascii=False) + '\n')


def write_file(content, filepath, mode='wt', encoding=None):
    """
    Write ``content`` to disk at ``filepath``. Files with appropriate extensions
    are compressed with gzip or bz2 automatically. Any intermediate folders
    not found on disk are automatically created.
    """
    with open_sesame(filepath, mode=mode, encoding=encoding) as f:
        f.write(content)


def write_file_lines(lines, filepath, mode='wt', encoding=None):
    """
    Write the content in ``lines`` to disk at ``filepath``, line by line. Files
    with appropriate extensions are compressed with gzip or bz2 automatically.
    Any intermediate folders not found on disk are automatically created.
    """
    with open_sesame(filepath, mode=mode, encoding=encoding) as f:
        for line in lines:
            f.write(line + '\n')


def write_spacy_docs(spacy_docs, filepath):
    """
    Serialize a sequence of ``spacy.Doc`` s to disk at ``filepath`` using Spacy's
    ``spacy.Doc.to_bytes()`` functionality.

    Args:
        spacy_docs (``spacy.Doc`` or iterable(``spacy.Doc``)): a single spacy doc
            or a sequence of spacy docs to serialize to disk at ``filepath``
        filepath (str): /path/to/file on disk from which spacy docs will be streamed
    """
    if isinstance(spacy_docs, SpacyDoc):
        spacy_docs = (spacy_docs,)
    with open_sesame(filepath, mode='wb') as f:
        for doc in spacy_docs:
            f.write(doc.to_bytes())


def write_sparse_matrix(matrix, filepath, compressed=False):
    """
    Write a ``scipy.sparse.csr_matrix`` or ``scipy.sparse.csc_matrix`` to disk
    at ``filepath``, optionally compressed.

    Args:
        matrix (``scipy.sparse.csr_matrix`` or ``scipy.sparse.csr_matrix``)
        filepath (str): /path/to/file on disk to which matrix objects will be written;
            if ``filepath`` does not end in ``.npz``, that extension is
            automatically appended to the name

    .. See also: http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.savez.html
    """
    if not isinstance(matrix, (csc_matrix, csr_matrix)):
        raise TypeError('input matrix must be a scipy sparse csr or csc matrix')
    make_dirs(filepath, 'w')
    if compressed is False:
        savez(filepath,
              data=matrix.data, indices=matrix.indices,
              indptr=matrix.indptr, shape=matrix.shape)
    else:
        savez_compressed(filepath,
                         data=matrix.data, indices=matrix.indices,
                         indptr=matrix.indptr, shape=matrix.shape)


def write_conll(spacy_doc, filepath, encoding=None):
    """
    Convert a single ``spacy.Doc`` into CoNLL-U format, and save it to disk.

    Args:
        spacy_doc (``spacy.Doc``): must be parsed
        filepath (str): /path/to/file on disk to which CONLL string will be written
        encoding (str, optional)

    Notes:
        See http://universaldependencies.org/docs/format.html for details.
    """
    rows = []
    for j, sent in enumerate(spacy_doc.sents):
        sent_i = sent.start
        sent_id = j + 1
        rows.append('# sent_id {}'.format(sent_id))
        for i, tok in enumerate(sent):
            # HACK...
            if tok.is_space:
                form = ' '
                lemma = ' '
            else:
                form = tok.orth_
                lemma = tok.lemma_
            tok_id = i + 1
            head = tok.head.i - sent_i + 1
            if head == tok_id:
                head = 0
            misc = 'SpaceAfter=No' if not tok.whitespace_ else '_'
            rows.append('\t'.join([str(tok_id), form, lemma, tok.pos_, tok.tag_,
                                   '_', str(head), tok.dep_.lower(), '_', misc]))
        rows.append('')  # sentences must be separated by a single newline
    conll = '\n'.join(rows)
    write_file(conll, filepath, mode='wt', encoding=encoding)
