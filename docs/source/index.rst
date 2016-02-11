.. textacy documentation master file, created by
   sphinx-quickstart on Thu Feb  4 13:08:20 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


textacy: higher-level NLP built on Spacy
========================================

``textacy`` is a Python library for performing higher-level natural language processing (NLP) tasks, built on the high-performance `Spacy <https://spacy.io/>`_ library. With the basics --- tokenization, part-of-speech tagging, parsing --- offloaded to another library, ``textacy`` focuses on tasks facilitated by the availability of tokenized, POS-tagged, and parsed text: keyterm extraction, readability statistics, emotional valence analysis, quotation attribution, and more.


Features
--------

- Functions for preprocessing raw text prior to analysis (whitespace normalization, URL/email/number/date replacement, unicode fixing/stripping, etc.)
- Convenient interface to basic linguistic elements provided by Spacy (words, ngrams, noun phrases, etc.), along with standardized filtering options
- Variety of functions for extracting information from text (particular POS patterns, subject-verb-object triples, acronyms and their definitions, direct quotations, etc.)
- Unsupervised key term extraction (specific algorithms such as SGRank or TextRank, as well as a general semantic network-based approach)
- Conversion of individual documents into common representations (bag of words), as well as corpora (term-document matrix, with TF or TF-IDF weighting, and filtering by these metrics or IC)
- Common utility functions for identifying a text's language, displaying key words in context (KWIC), truecasing words, and higher-level navigation of a parse tree

And more!


Installation
------------

.. code-block:: bash

    $ pip install -U textacy

^ VERY SOON.


Example
-------

.. code-block:: python

    >>> from textacy import TextDoc
    >>> with open('./resources/example.txt') as f:
    >>>     text = f.read()
    >>> metadata = {'title': 'What 74 Years of Crossword History Says About the Language We Use',
                    'publisher': 'NYTimes',
                    'url': 'http://www.nytimes.com/interactive/2016/02/07/opinion/what-74-years-of-times-crosswords-say-about-the-words-we-use.html',
                    'publish_date': '2016-02-06'}
    >>> doc = TextDoc(text, metadata=metadata)
    >>> print(doc)
    TextDoc(1667 tokens)
    >>> doc.named_entities(bad_ne_types='numeric',
                           min_freq=2, drop_determiners=True)[:10]
    [Americans,
     New York Times,
     Times,
     United States,
     English,
     Americans,
     German,
     Latin,
     Korean,
     United States]
    >>> doc.key_terms(algorithm='textrank', n=10)
    [('puzzle', 0.03221765466554432),
     ('word', 0.02142022685357814),
     ('language', 0.019210081575257998),
     ('clue', 0.012616820406126519),
     ('answer', 0.01192589432106701),
     ('crossword', 0.0106710664952726),
     ('year', 0.010411868535475188),
     ('international', 0.010295554570385957),
     ('vietnamese', 0.009281694480421225),
     ('spanish', 0.009250784588045964)]
    >>> doc.readability_stats()
    {'automated_readability_index': 11.898600326151069,
     'coleman_liau_index': 16.769160473613,
     'flesch_kincaid_grade_level': 8.910683355886334,
     'flesch_readability_ease': 48.05235514034905,
     'gunning_fog_index': 10.609770653343048,
     'n_chars': 4486,
     'n_polysyllable_words': 126,
     'n_sents': 78,
     'n_syllables': 1303,
     'n_unique_words': 447,
     'n_words': 739,
     'smog_index': 10.389873798559362}


Table of Contents
-----------------

.. toctree::
   :maxdepth: 2

   license
   api_reference
