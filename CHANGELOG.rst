Changelog
=========

Latest (WIP)
------------

Changes:

- Added ``corpora/bernie_and_hillary.py`` module, which handles downloading to and loading from disk a corpus of congressional speeches by Bernie Sanders and Hillary Clinton
- Modified ``data.load_depechemood`` function, now downloads data from GitHub source if not found on disk
- Removed ``resources/`` directory from GitHub, hence all the downloading

0.2.0 (2016-04-11)
------------------

Changes:

- Added ``representations`` subpackage; includes modules for network and vector space model (VSM) document and corpus representations
    - Document-term matrix creation now takes documents represented as a list of terms (rather than as spaCy Docs); splits the tokenization step from vectorization for added flexibility
    - Some of this functionality was refactored from existing parts of the package
- Added ``tm`` (topic modeling) subpackage, with a main ``TopicModel`` class for training, applying, persisting, and interpreting NMF, LDA, and LSA topic models through a single interface
- Various improvements to ``TextDoc`` and ``TextCorpus`` classes
    - ``TextDoc`` can now be initialized from a spaCy Doc
    - Removed caching from ``TextDoc``, because it was a pain and weird and probably not all that useful
    - ``extract``-based methods are now generators, like the functions they wrap
    - Added ``.as_semantic_network()`` and ``.as_terms_list()`` methods to ``TextDoc``
    - ``TextCorpus.from_texts()`` now takes advantage of multithreading via spaCy, if available, and document metadata can be passed in as a paired iterable of dicts
- Added read/write functions for sparse scipy matrices
- Added ``fileio.read.split_content_and_metadata()`` convenience function for splitting (text) content from associated metadata when reading data from disk into a ``TextDoc`` or ``TextCorpus``
- Renamed ``fileio.read.get_filenames_in_dir()`` to ``fileio.read.get_filenames()`` and added functionality for matching/ignoring files by their names, file extensions, and ignoring invisible files
- Rewrote ``export.docs_to_gensim()``, now significantly faster
- Imports in ``__init__.py`` files for main and subpackages now explicit

Bugfixes:

- ``textstats.readability_stats()`` no longer filters out stop words (@henningko #7)
- Wikipedia article processing now recursively removes nested markup
- ``extract.ngrams()`` now filters out ngrams with any space-only tokens
- functions with ``include_nps`` kwarg changed to ``include_ncs``, to match the renaming of the associated function from ``extract.noun_phrases()`` to ``extract.noun_chunks()``

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
