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

- Access spaCy through convenient methods for working with one or many documents
  and extend its functionality through custom extensions and automatic language
  identification for applying the right spaCy pipeline for the text
- Download datasets with both text content and metadata, from Congressional speeches
  to historical literature to Reddit comments
- Easily stream data to and from disk in many common formats
- Clean, normalize, and explore raw text — before processing it with spaCy
- Flexibly extract words, n-grams, noun chunks, entities, acronyms, key terms,
  and other elements of interest from processed documents
- Compare strings, sets, and documents by a variety of similarity metrics
- Tokenize and vectorize documents then train, interpret, and visualize topic models
- Compute a variety of text readability statistics, including Flesch-Kincaid grade level,
  SMOG index, and multi-lingual Flesch Reading Ease

... *and more!*

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   api_reference/root
   license
