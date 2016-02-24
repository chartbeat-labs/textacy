"""
Module with functions for writing content to disk in common formats.
"""
import bz2
import gzip
import io
import json

import ijson


def write_json(json_object, filename, mode='wt', encoding=None):
    """
    Write JSON object all at once to disk at ``filename``.

    Args:
        json_object (json): valid JSON object to be written
        filename (str): /path/to/file on disk to which json object will be written,
            such as a JSON array; for example::

                [
                    {"title": "Harrison Bergeron", "text": "The year was 2081, and everybody was finally equal."},
                    {"title": "2BR02B", "text": "Everything was perfectly swell."}
                ]

        mode (str, optional)
        encoding (str, optional)
    """
    with io.open(filename, mode=mode, encoding=encoding) as f:
        json.dump(json_object, f, ensure_ascii=False)


def write_json_lines(json_objects, filename, mode='wt', encoding=None):
    """
    Iterate over a stream of JSON objects, writing each to a separate line in
    file ``filename`` but without a top-level JSON object (e.g. array).

    Args:
        json_objects (iterable(json)): sequence of valid JSON objects to be written
        filename (str): /path/to/file on disk to which JSON objects will be written,
            where each line in the file is its own json object; for example::

                {"title": "Harrison Bergeron", "text": "The year was 2081, and everybody was finally equal."}\n
                {"title": "2BR02B", "text": "Everything was perfectly swell."}

        mode (str, optional)
        encoding (str, optional)
    """
    with io.open(filename, mode=mode, encoding=encoding) as f:
        for json_object in json_objects:
            f.write(json.dumps(json_object, ensure_ascii=False) + '\n')


def write_json_mash(json_objects, filename, mode='wt', encoding=None):
    """
    Iterate over a stream of JSON objects, writing each to a single line in
    file ``filename`` but without a top-level JSON object (e.g. array). This is
    bad form; you should probably use :func:`write_json_lines <textacy.fileio.write.write_json_lines>`

    Args:
        json_objects (iterable(json)): sequence of valid JSON objects to be written
        filename (str): /path/to/file on disk to which JSON objects will be
            written together, end-to-end, on a single line; for example::

                {"title": "Harrison Bergeron", "text": "The year was 2081, and everybody was finally equal."}{"title": "2BR02B", "text": "Everything was perfectly swell."}

        mode (str, optional)
        encoding (str, optional)
    """
    with io.open(filename, mode=mode, encoding=encoding) as f:
        for json_object in json_objects:
            f.write(json.dumps(json_object, ensure_ascii=False))


def write_file(obj, filename, mode='wt', encoding=None):
    """
    Write content of ``obj`` to disk at ``filename``. Files with appropriate
    extensions are compressed with gzip or bz2 automatically.
    """
    _open = gzip.open if filename.endswith('.gz') \
        else bz2.open if filename.endswith('.bz2') \
        else io.open
    with _open(filename, mode=mode, encoding=encoding) as f:
        f.write(obj)


def write_file_lines(objects, filename, mode='wt', encoding=None):
    """
    Write the objects in ``objects`` to disk at ``filename``, line by line. Files
    with appropriate extensions are compressed with gzip or bz2 automatically.
    """
    _open = gzip.open if filename.endswith('.gz') \
        else bz2.open if filename.endswith('.bz2') \
        else io.open
    with _open(filename, mode=mode, encoding=encoding) as f:
        for obj in objects:
            f.write(obj +'\n')


def write_spacy_docs(spacy_docs, filename):
    """
    Serialize a sequence of ``spacy.Doc`` s to disk at ``filename`` using Spacy's
    ``spacy.Doc.to_bytes()`` functionality.

    Args:
        spacy_docs (iterable(``spacy.Doc``)): sequence of spacy docs to serialize
            to disk at ``filename``
        filename (str): /path/to/file on disk from which spacy docs will be streamed
    """
    with io.open(filename, mode='wb') as f:
        for doc in spacy_docs:
            f.write(doc.to_bytes())
