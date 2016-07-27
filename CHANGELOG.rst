Changelog
=========

dev
---

Changes:

- refactored and improved `fileio` subpackage
    - moved shared (read/write) functions into separate `fileio.utils` module
    - almost all read/write functions now use `fileio.utils.open_sesame()`, enabling seamless fileio for uncompressed or gzip, bz2, and lzma compressed files; relative/user-home-based paths; and missing intermediate directories
    - added options for writing json files (matching stdlib's `json.dump()`) that can help save space
    - `fileio.utils.get_filenames()` now matches for/against a regex pattern rather than just a contained substring; using the old params will now raise a deprecation warning
    - BREAKING: `fileio.utils.split_content_and_metadata()` now has `itemwise=False` by default, rather than `itemwise=True`, which means that splitting multi-document streams of content and metadata into parallel iterators is now the default action
    - NOTE: certain file mode / compression pairs simply don't work (this is Python's fault), so users may run into exceptions; in Python 3, you'll almost always want to use text mode ('wt' or 'rt'), but in Python 2, users can't read or write compressed files in text mode, only binary mode ('wb' or 'rb')
- cleaned up deprecated/bad Py2/3 `compat` imports, and added better functionality for Py2/3 strings
    - now `compat.unicode_type` used for text data, `compat.bytes_type` for binary data, and `compat.string_types` for when either will do
    - also added `compat.unicode_to_bytes()` and `compat.bytes_to_unicode()` functions, for converting between string types
- added `compression` param to `TextCorpus.save()` and `.load()` to optionally write metadata json file in compressed form
- moved `fileio.write_conll()` functionality to `export.doc_to_conll()`, which converts a spaCy doc into a ConLL-U formatted string; writing that string to disk would require a separate call to `fileio.write_file()`

Bugfixes:

- Fixed document(s) removal from `TextCorpus` objects, including correct decrementing of `.n_docs`, `.n_sents`, and `.n_tokens` attributes (@michelleful #29)


0.2.5 (2016-07-14)
------------------

Bugfixes:

- Added (missing) `pyemd` and `python-levenshtein` dependencies to requirements and setup files
- Fixed bug in `data.load_depechemood()` arising from the Py2 `csv` module's inability to take unicode as input (thanks to @robclewley, issue #25)


0.2.4 (2016-07-14)
------------------

Changes:

- New features for `TextDoc` and `TextCorpus` classes
    - added `.save()` methods and `.load()` classmethods, which allows for fast serialization of parsed documents/corpora and associated metadata to/from disk — with an important caveat: if `spacy.Vocab` object used to serialize and deserialize is not the same, there will be problems, making this format useful as short-term but not long-term storage
    - `TextCorpus` may now be instantiated with an already-loaded spaCy pipeline, which may or may not have all models loaded; it can still be instantiated using a language code string ('en', 'de') to load a spaCy pipeline that includes all models by default
    - `TextDoc` methods wrapping `extract` and `keyterms` functions now have full documentation rather than forwarding users to the wrapped functions themselves; more irritating on the dev side, but much less irritating on the user side :)
- Added a `distance.py` module containing several document, set, and string distance metrics
    - word movers: document distance as distance between individual words represented by word2vec vectors, normalized
    - "word2vec": token, span, or document distance as cosine distance between (average) word2vec representations, normalized
    - jaccard: string or set(string) distance as intersection / overlap, normalized, with optional fuzzy-matching across set members
    - hamming: distance between two strings as number of substititions, optionally normalized
    - levenshtein: distance between two strings as number of substitions, deletions, and insertions, optionally normalized (and removed a redundant function from the still-orphaned `math_utils.py` module)
    - jaro-winkler: distance between two strings with variable prefix weighting, normalized
- Added `most_discriminating_terms()` function to `keyterms` module to take a collection of documents split into two exclusive groups and compute the most discriminating terms for group1-and-not-group2 as well as group2-and-not-group1

Bugfixes:

- fixed variable name error in docs usage example (thanks to @licyeus, PR #23)


0.2.3 (2016-06-20)
------------------

Changes:

- Added `corpora.RedditReader()` class for streaming Reddit comments from disk, with `.texts()` method for a stream of plaintext comments and `.comments()` method for a stream of structured comments as dicts, with basic filtering by text length and limiting the number of comments returned
- Refactored functions for streaming Wikipedia articles from disk into a `corpora.WikiReader()` class, with `.texts()` method for a stream of plaintext articles and `.pages()` method for a stream of structured pages as dicts, with basic filtering by text length and limiting the number of pages returned
- Updated README and docs with a more comprehensive — and correct — usage example; also added tests to ensure it doesn't get stale
- Updated requirements to latest version of spaCy, as well as added matplotlib for `viz`

Bugfixes:

- `textacy.preprocess.preprocess_text()` is now, once again, imported at the top level, so easily reachable via `textacy.preprocess_text()` (@bretdabaker #14)
- `viz` subpackage now included in the docs' API reference
- missing dependencies added into `setup.py` so pip install handles everything for folks


0.2.2 (2016-05-05)
------------------

Changes:

- Added a `viz` subpackage, with two types of plots (so far):
    - `viz.draw_termite_plot()`, typically used to evaluate and interpret topic models; conveniently accessible from the `tm.TopicModel` class
    - `viz.draw_semantic_network()` for visualizing networks such as those output by `representations.network`
- Added a "Bernie & Hillary" corpus with 3000 congressional speeches made by Bernie Sanders and Hillary Clinton since 1996
    - ``corpora.fetch_bernie_and_hillary()`` function automatically downloads to and loads from disk this corpus
- Modified ``data.load_depechemood`` function, now downloads data from GitHub source if not found on disk
- Removed ``resources/`` directory from GitHub, hence all the downloadin'
- Updated to spaCy v0.100.7
    - German is now supported! although some functionality is English-only
    - added `textacy.load_spacy()` function for loading spaCy packages, taking advantage of the new `spacy.load()` API; added a DeprecationWarning for `textacy.data.load_spacy_pipeline()`
    - proper nouns' and pronouns' ``.pos_`` attributes are now correctly assigned 'PROPN' and 'PRON'; hence, modified ``regexes_etc.POS_REGEX_PATTERNS['en']`` to include 'PROPN'
    - modified ``spacy_utils.preserve_case()`` to check for language-agnostic 'PROPN' POS rather than English-specific 'NNP' and 'NNPS' tags
- Added `text_utils.clean_terms()` function for cleaning up a sequence of single- or multi-word strings by stripping leading/trailing junk chars, handling dangling parens and odd hyphenation, etc.

Bugfixes:

- ``textstats.readability_stats()`` now correctly gets the number of words in a doc from its generator function (@gryBox #8)
- removed NLTK dependency, which wasn't actually required
- ``text_utils.detect_language()`` now warns via ``logging`` rather than a ``print()`` statement
- ``fileio.write_conll()`` documentation now correctly indicates that the filename param is not optional


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
