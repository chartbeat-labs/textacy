Changelog
=========

dev
---

Changes:

- Added `.save()` methods and `.load()` classmethods to both `TextDoc` and `TextCorpus` classes, which allows for fast serialization of parsed documents and associated metadata to/from disk.
    - caveat: if `spacy.Vocab` object used to serialize and deserialize is not the same, there will be problems, making this format useful as short-term but not long-term storage
- `TextCorpus` may now be instantiated with an already-loaded spaCy pipeline, which may or may not have all models loaded; it can still be instantiated using a language code string ('en', 'de') to load a spaCy pipeline that includes all models by default
- Added a `distance.py` module containing several document, set, and string distance metrics
    - word movers: document distance as distance between individual words represented by word2vec vectors, normalized
    - "word2vec": token, span, or document distance as cosine distance between (average) word2vec representations, normalized
    - jaccard: string or set(string) distance as intersection / overlap, normalized, with optional fuzzy-matching across set members
    - hamming: distance between two strings as number of substititions, optionally normalized
    - levenshtein: distance between two strings as number of substitions, deletions, and insertions, optionally normalized (and removed a redundant function from the still-orphaned `math_utils.py` module)
    - jaro-winkler: distance between two strings with variable prefix weighting, normalized

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
