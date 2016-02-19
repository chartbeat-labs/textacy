Changelog
=========

0.2.0 (WIP)
-----------

Changes:

- Added ``export.py`` module for exporting textacy/spacy objects into "third-party" formats; so far, just gensim and conll-u
- Added ``compat.py`` module for Py2/3 compatibility hacks

Bug fixes:

- Whitespace tokens now always filtered out of ``extract.words()`` lists
- Some Py2/3 str/unicode issues fixed
