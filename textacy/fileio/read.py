"""
Module with functions for reading content from disk in common formats.
"""
import bz2
from functools import partial
import gzip
import io
import json
import os

import ijson
from spacy.tokens.doc import Doc


JSON_DECODER = json.JSONDecoder()


def read_json(filename, mode='rt', encoding=None, prefix=''):
    """
    Iterate over JSON objects matching the field given by ``prefix``.
    Useful for reading a large JSON array one item (with ``prefix='item'``)
    or sub-item (``prefix='item.fieldname'``) at a time.

    Args:
        filename (str): /path/to/file on disk from which json items will be streamed,
            such as items in a JSON array; for example::

                [
                    {"title": "Harrison Bergeron", "text": "The year was 2081, and everybody was finally equal."},
                    {"title": "2BR02B", "text": "Everything was perfectly swell."}
                ]

        mode (str, optional)
        encoding (str, optional)
        prefix (str, optional): if '', the entire JSON object will be read in at once;
            if 'item', each item in a top-level array will be read in successively;
            if 'item.text', each array item's 'text' value will be read in successively

    Yields:
        next matching JSON object; could be a dict, list, int, float, str,
            depending on the value of ``prefix``

    Notes:
        Refer to ``ijson`` at https://pypi.python.org/pypi/ijson/ for usage details.
    """
    with io.open(filename, mode=mode, encoding=encoding) as f:
        if prefix == '':
            yield json.load(f)
        else:
            for item in ijson.items(f, prefix):
                yield item


def read_json_lines(filename, mode='rt', encoding=None):
    """
    Iterate over a stream of JSON objects, where each line of file ``filename``
    is a valid JSON object but no JSON object (e.g. array) exists at the top level.

    Args:
        filename (str): /path/to/file on disk from which json objects will be streamed,
            where each line in the file must be its own json object; for example::

                {"title": "Harrison Bergeron", "text": "The year was 2081, and everybody was finally equal."}\n
                {"title": "2BR02B", "text": "Everything was perfectly swell."}

        mode (str, optional)
        encoding (str, optional)

    Yields:
        dict: next valid JSON object, converted to native Python equivalent
    """
    with io.open(filename, mode=mode, encoding=encoding) as f:
        for line in f:
            yield json.loads(line)


def read_json_mash(filename, mode='rt', encoding=None, buffersize=2048):
    """
    Iterate over a stream of JSON objects, all of them mashed together, end-to-end,
    on a single line of a file. Bad form, but still manageable.

    Args:
        filename (str): /path/to/file on disk from which json objects will be streamed,
            where all json objects are mashed together, end-to-end, on a single line,;
            for example::

                {"title": "Harrison Bergeron", "text": "The year was 2081, and everybody was finally equal."}{"title": "2BR02B", "text": "Everything was perfectly swell."}

        mode (str, optional)
        encoding (str, optional)
        buffersize (int, optional): number of bytes to read in as a chunk

    Yields:
        dict: next valid JSON object, converted to native Python equivalent
    """
    with io.open(filename, mode=mode, encoding=encoding) as f:
        buffer = ''
        for chunk in iter(partial(f.read, buffersize), ''):
            buffer += chunk
            while buffer:
                try:
                    result, index = JSON_DECODER.raw_decode(buffer)
                    yield result
                    buffer = buffer[index:]
                # not enough data to decode => read another chunk
                except ValueError:
                    break


def read_file(filename, mode='rt', encoding=None):
    """
    Read the full contents of a file. Files compressed with gzip or bz2 are handled
    automatically.
    """
    _open = gzip.open if filename.endswith('.gz') \
        else bz2.open if filename.endswith('.bz2') \
        else io.open
    with _open(filename, mode=mode, encoding=encoding) as f:
        return f.read()


def read_file_lines(filename, mode='rt', encoding=None):
    """
    Read the contents of a file, line by line. Files compressed with gzip or bz2
    are handled automatically.
    """
    _open = gzip.open if filename.endswith('.gz') \
        else bz2.open if filename.endswith('.bz2') \
        else io.open
    with _open(filename, mode=mode, encoding=encoding) as f:
        for line in f:
            yield f


def read_spacy_docs(spacy_vocab, filename):
    """
    Stream ``spacy.Doc`` s from disk at ``filename`` where they were serialized
    using Spacy's ``spacy.Doc.to_bytes()`` functionality.

    Args:
        spacy_vocab (``spacy.Vocab``): the spacy vocab object used to serialize
            the docs in ``filename``
        filename (str): /path/to/file on disk from which spacy docs will be streamed

    Yields:
        the next deserialized ``spacy.Doc``
    """
    with io.open(filename, mode='rb') as f:
        for bytes_string in Doc.read_bytes(f):
            yield Doc(spacy_vocab).from_bytes(bytes_string)


def get_filenames_in_dir(dirname, file_type=None, subdirs=False):
    """
    Yield the full names of files under directory ``dirname``, optionally
    filtering for a certain file type and crawling all subdirectories.

    Args:
        dirname (str): /path/to/dir on disk where files to read are saved
        file_type (str, optional): if files only of a certain type are wanted,
            specify the file extension (e.g. ".txt")
        subdirs (bool, optional): if True, iterate through all files in subdirectories

    Yields:
        str: next file's name, including the full path on disk

    Raises:
        IOError: if ``dirname`` is not found on disk
    """
    if not os.path.exists(dirname):
        raise IOError('directory {} does not exist'.format(dirname))

    for dirpath, dirnames, filenames in os.walk(dirname):
        if dirpath.startswith('.'):
            continue
        for filename in filenames:
            if filename.startswith('.'):
                continue
            if file_type and not file_type.endswith(file_type):
                continue
            yield os.path.join(dirpath, filename)

        if subdirs is False:
            break
