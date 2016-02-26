Changelog
=========

0.1.4 (2016-02-26)
------------------

Changes:

- Added ``corpora`` subpackage with ``wikipedia.py`` module; functions for streaming pages from a Wikipedia db dump as plain text or structured data
- Added ``fileio`` subpackage with functions for reading/writing content from/to disk in common formats
  - JSON formats, both standard and streaming-friendly
  - text, optionally compressed
  - spacy documents to/from binary

0.1.3 (2016-02-22)
------------------

Changes:

- Added ``export.py`` module for exporting textacy/spacy objects into "third-party" formats; so far, just gensim and conll-u
- Added ``compat.py`` module for Py2/3 compatibility hacks
- Renamed ``extract.noun_phrases()`` to ``extract.noun_chunks()`` to match Spacy's API
- Changed extract functions to generators, rather than returning lists
- Added ``TextDoc.merge()`` and ``spacy_utils.merge_spans()`` for merging spans into single tokens within a ``spacy.Doc``, uses Spacy's recent implementation

Bug fixes:

- Whitespace tokens now always filtered out of ``extract.words()`` lists
- Some Py2/3 str/unicode issues fixed
- Broken tests in ``test_extract.py`` no longer broken
