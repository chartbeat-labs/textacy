"""
Module with functions for writing content to disk in common formats.
"""
import bz2
import gzip
import io
import json

import ijson
from spacy.tokens.doc import Doc as SpacyDoc


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


def write_file(content, filename, mode='wt', encoding=None):
    """
    Write ``content`` to disk at ``filename``. Files with appropriate extensions
    are compressed with gzip or bz2 automatically.
    """
    _open = gzip.open if filename.endswith('.gz') \
        else bz2.open if filename.endswith('.bz2') \
        else io.open
    with _open(filename, mode=mode, encoding=encoding) as f:
        f.write(content)


def write_file_lines(lines, filename, mode='wt', encoding=None):
    """
    Write the content in ``lines`` to disk at ``filename``, line by line. Files
    with appropriate extensions are compressed with gzip or bz2 automatically.
    """
    _open = gzip.open if filename.endswith('.gz') \
        else bz2.open if filename.endswith('.bz2') \
        else io.open
    with _open(filename, mode=mode, encoding=encoding) as f:
        for line in lines:
            f.write(line +'\n')


def write_spacy_docs(spacy_docs, filename, encoding=None):
    """
    Serialize a sequence of ``spacy.Doc`` s to disk at ``filename`` using Spacy's
    ``spacy.Doc.to_bytes()`` functionality.

    Args:
        spacy_docs (``spacy.Doc`` or iterable(``spacy.Doc``)): a single spacy doc
            or a sequence of spacy docs to serialize to disk at ``filename``
        filename (str): /path/to/file on disk from which spacy docs will be streamed
        encoding (str, optional)
    """
    if isinstance(spacy_docs, SpacyDoc):
        spacy_docs = [spacy_docs]
    with io.open(filename, mode='wb') as f:
        for doc in spacy_docs:
            f.write(doc.to_bytes())


def write_conll(spacy_doc, filename, encoding=None):
    """
    Convert a single ``spacy.Doc`` into CoNLL-U format, and save it to disk.

    Args:
        spacy_doc (``spacy.Doc``): must be parsed
        filename (str, optional): to save the CoNLL string to disk, provide the full
            path/to/fname.txt; otherwise, the string is returned but not saved
        encoding (str, optional)

    Notes:
        See http://universaldependencies.org/docs/format.html for details.
    """
    rows = []
    for j, sent in enumerate(doc.sents):
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
    write_file(conll, filename, mode='wt', encoding=encoding)
