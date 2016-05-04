========================================
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
- Sklearn-style topic modeling with LSA, LDA, or NMF, including functions to interpret the results of trained models

And more!


Installation
------------

The simple way to install ``textacy`` is

.. code-block:: bash

    $ pip install -U textacy

Or, download and unzip the source ``tar.gz`` from  `PyPi <https://pypi.python.org/pypi/textacy>`_, then

.. code-block:: bash

    $ python setup.py install


Example
-------

.. code-block:: pycon

    >>> import textacy
    >>>
    >>> text = """
    ... Hell, it's about time someone told about my friend EPICAC. After all, he cost the taxpayers $776,434,927.54. They have a right to know about him, picking up a check like that. EPICAC got a big send off in the papers when Dr. Ormand von Kleigstadt designed him for the Government people. Since then, there hasn't been a peep about him -- not a peep. It isn't any military secret about what happened to EPICAC, although the Brass has been acting as though it were. The story is embarrassing, that's all. After all that money, EPICAC didn't work out the way he was supposed to.
    ... And that's another thing: I want to vindicate EPICAC. Maybe he didn't do what the Brass wanted him to, but that doesn't mean he wasn't noble and great and brilliant. He was all of those things. The best friend I ever had, God rest his soul.
    ... You can call him a machine if you want to. He looked like a machine, but he was a whole lot less like a machine than plenty of people I could name. That's why he fizzled as far as the Brass was concerned.
    ... """
    >>> textacy.preprocess_text(text, lowercase=True, no_numbers=True, no_punct=True)
    'hell its about time someone told about my friend epicac after all he cost the taxpayers number they have a right to know about him picking up a check like that epicac got a big send off in the papers when dr ormand von kleigstadt designed him for the government people since then there hasnt been a peep about him not a peep it isnt any military secret about what happened to epicac although the brass has been acting as though it were the story is embarrassing thats all after all that money epicac didnt work out the way he was supposed to\nand thats another thing i want to vindicate epicac maybe he didnt do what the brass wanted him to but that doesnt mean he wasnt noble and great and brilliant he was all of those things the best friend i ever had god rest his soul\nyou can call him a machine if you want to he looked like a machine but he was a whole lot less like a machine than plenty of people i could name thats why he fizzled as far as the brass was concerned'
    >>> textacy.text_utils.keyword_in_context(text, 'EPICAC', window_width=40)
    about time someone told about my friend  EPICAC . After all, he cost the taxpayers $776,
    bout him, picking up a check like that.  EPICAC  got a big send off in the papers when D
     military secret about what happened to  EPICAC , although the Brass has been acting as
    sing, that's all. After all that money,  EPICAC  didn't work out the way he was supposed
    at's another thing: I want to vindicate  EPICAC . Maybe he didn't do what the Brass want
    >>>
    >>> doc = textacy.TextDoc(text.strip(), lang='auto',
    ...                       metadata={'title': 'EPICAC', 'author': 'Kurt Vonnegut'})
    >>> print(doc)
    TextDoc(230 tokens)
    >>> doc.lang
    'en'
    >>>
    >>> doc.ngrams(2, filter_stops=True, filter_punct=True)
    [friend EPICAC.,
     taxpayers $,
     $776,434,927.54,
     check like,
     EPICAC got,
     big send,
     Dr. Ormand,
     Ormand von,
     von Kleigstadt,
     Kleigstadt designed,
     Government people,
     military secret,
     n't work,
     vindicate EPICAC.,
     EPICAC. Maybe,
     Brass wanted,
     n't mean,
     n't noble,
     best friend,
     God rest,
     looked like]
    >>> doc.ngrams(3, filter_stops=True, filter_punct=True, min_freq=2)
    [like a machine, like a machine]
    >>> doc.named_entities(drop_determiners=True, bad_ne_types='numeric')
    [Hell, EPICAC, Ormand von Kleigstadt, EPICAC, EPICAC, Brass, God]
    >>> doc.pos_regex_matches(r'<DET> <NUM>* (<ADJ> <PUNCT>? <CONJ>?)* (<NOUN> <PART>?)+')
    [the taxpayers,
     a right to,
     a check,
     the papers,
     the Government people,
     a peep,
     a peep,
     any military secret,
     the Brass,
     The story,
     that money,
     the way he,
     another thing,
     the Brass,
     those things,
     The best friend I,
     a machine,
     a machine,
     a whole lot,
     a machine,
     the Brass]
    >>> doc.semistructured_statements('he', cue='be')
    [(he, was, n't noble and great and brilliant),
     (He, was, all of those things),
     (he, was, a whole lot less like a machine than plenty of people I could name)]
    >>> doc.key_terms(algorithm='textrank', n=5)
    [('EPICAC', 0.06369346448602185),
     ('Brass', 0.051763452142722675),
     ('machine', 0.04761999319651037),
     ('friend', 0.045713561400759786),
     ('people', 0.043303827328545416)]
    >>> doc.readability_stats()
    {'automated_readability_index': 5.848928571428573,
     'coleman_liau_index': 9.577214607142864,
     'flesch_kincaid_grade_level': 3.7476190476190503,
     'flesch_readability_ease': 78.8807142857143,
     'gunning_fog_index': 4.780952380952381,
     'n_chars': 433,
     'n_polysyllable_words': 5,
     'n_sents': 14,
     'n_syllables': 121,
     'n_unique_words': 62,
     'n_words': 84,
     'smog_index': 6.5431188927421005}
    >>> doc.term_count('EPICAC')
    3
    >>> bot = doc.as_bag_of_terms(weighting='tf', normalized=False,
    ...                           lemmatize='auto', ngram_range=(1, 2))
    >>> [(doc.spacy_stringstore[term_id], count)
    ...  for term_id, count in bot.most_common(n=10)]
    [('not', 6),
     ("'", 4),
     ('EPICAC', 3),
     ('want', 3),
     ('Brass', 3),
     ('like', 3),
     ('machine', 3),
     ('\n', 2),
     ('people', 2),
     ('friend', 2)]


Project Links
-------------

- `textacy @ PyPi <https://pypi.python.org/pypi/textacy>`_
- `textacy @ GitHub <https://github.com/chartbeat-labs/textacy>`_
- `textacy @ ReadTheDocs <http://textacy.readthedocs.io/en/latest/>`_


Table of Contents
-----------------

.. toctree::
   :maxdepth: 2

   license
   api_reference
