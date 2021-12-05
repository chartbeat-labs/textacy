textacy: NLP, before and after spaCy
====================================

``textacy`` is a Python library for performing a variety of natural language processing (NLP)
tasks, built on the high-performance spaCy library. With the fundamentals --- tokenization,
part-of-speech tagging, dependency parsing, etc. --- delegated to another library,
``textacy`` focuses primarily on the tasks that come before and follow after.

.. image:: https://img.shields.io/travis/chartbeat-labs/textacy/master.svg?style=flat-square
   :target: https://travis-ci.org/chartbeat-labs/textacy
   :alt: build status

.. image:: https://img.shields.io/github/release/chartbeat-labs/textacy.svg?style=flat-square
   :target: https://github.com/chartbeat-labs/textacy/releases
   :alt: current release version

.. image:: https://img.shields.io/pypi/v/textacy.svg?style=flat-square
   :target: https://pypi.python.org/pypi/textacy
   :alt: pypi version

.. image:: https://anaconda.org/conda-forge/textacy/badges/version.svg
   :target: https://anaconda.org/conda-forge/textacy
   :alt: conda version

features
--------

- Access and extend spaCy's core functionality for working with one or many documents
  through convenient methods and custom extensions
- Load prepared datasets with both text content and metadata, from Congressional speeches
  to historical literature to Reddit comments
- Clean, normalize, and explore raw text before processing it with spaCy
- Extract structured information from processed documents, including n-grams, entities,
  acronyms, keyterms, and SVO triples
- Compare strings and sequences using a variety of similarity metrics
- Tokenize and vectorize documents then train, interpret, and visualize topic models
- Compute text readability and lexical diversity statistics, including Flesch-Kincaid
  grade level, multilingual Flesch Reading Ease, and Type-Token Ratio

... *and much more!*

links
-----

- Download: https://pypi.org/project/textacy
- Documentation: https://textacy.readthedocs.io
- Source code: https://github.com/chartbeat-labs/textacy
- Bug Tracker: https://github.com/chartbeat-labs/textacy/issues

maintainer
----------

Howdy, y’all. 👋

- Burton DeWilde (burtdewilde@gmail.com)

contents
--------

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   walkthrough
   tutorials/root
   api_reference/root
   changes
