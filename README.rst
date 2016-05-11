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

.. code-block:: console

    $ pip install -U textacy

Or, download and unzip the source ``tar.gz`` from  `PyPi <https://pypi.python.org/pypi/textacy>`_, then

.. code-block:: console

    $ python setup.py install


Example
-------

.. code-block:: pycon

    >>> import textacy

Efficiently stream documents from disk and into a processed corpus:

.. code-block:: pycon

    >>> docs = textacy.corpora.fetch_bernie_and_hillary()
    >>> content_stream, metadata_stream = textacy.fileio.split_content_and_metadata(
    ...     docs, 'text', itemwise=False)
    >>> corpus = textacy.TextCorpus.from_texts(
    ...     'en', content_stream, metadata_stream, n_threads=2)
    >>> print(corpus)
    TextCorpus(3066 docs; 1909705 tokens)

Represent corpus as a document-term matrix, with flexible weighting and filtering:

.. code-block:: pycon

    >>> doc_term_matrix, id2term = corpus.as_doc_term_matrix(
    ...     (doc.as_terms_list(words=True, ngrams=False, named_entities=True)
    ...      for doc in corpus),
    ...     weighting='tfidf', normalize=True, smooth_idf=True, min_df=2, max_df=0.95)
    >>> print(repr(doc_term_matrix))
    <3066x16145 sparse matrix of type '<class 'numpy.float64'>'
    	with 432067 stored elements in Compressed Sparse Row format>

Train and interpret a topic model:

.. code-block:: pycon

    >>> model = textacy.tm.TopicModel('nmf', n_topics=10)
    >>> model.fit(doc_term_matrix)
    >>> doc_topic_matrix = model.transform(doc_term_matrix)
    >>> print(doc_topic_matrix.shape)
    (3066, 10)
    >>> for topic_idx, top_terms in model.top_topic_terms(id2term, top_n=10):
    ...     print('topic', topic_idx, ':', '   '.join(top_terms))
    topic 0 : people   tax   $   percent   american   million   republican   country   go   americans
    topic 1 : rescind   quorum   order   consent   unanimous   ask   president   mr.   madam   absence
    topic 2 : chairman   chairman.   amendment   mr.   clerk   gentleman   designate   offer   sanders   vermont
    topic 3 : dispense   reading   amendment   consent   unanimous   ask   president   mr.   madam   pending
    topic 4 : senate   consent   session   unanimous   authorize   ask   committee   meet   president   a.m.
    topic 5 : health   care   state   child   veteran   va   vermont   new   's   need
    topic 6 : china   american   speaker   worker   trade   job   wage   america   gentleman   people
    topic 7 : social security   social   security   cut   senior   medicare   deficit   benefit   year   cola
    topic 8 : senators   desiring   chamber   vote   minute   morning   permit   10 minute   proceed   speak
    topic 9 : motion   table   reconsider   lay   agree   preamble   record   resolution   consent   print

Basic indexing as well as flexible selection of documents in a corpus:

.. code-block:: pycon

    >>> bernie_docs = list(corpus.get_docs(
    ...     lambda doc: doc.metadata['speaker'] == 'Bernard Sanders'))
    >>> print(len(bernie_docs))
    2236
    >>> doc = corpus[-1]
    >>> print(doc)
    TextDoc(465 tokens; "Mr. President, I ask to have printed in the Rec...")

Preprocess plain text, or highlight particular terms in it:

.. code-block:: pycon

    >>> textacy.preprocess_text(doc.text, lowercase=True, no_punct=True)[:70]
    'mr president i ask to have printed in the record copies of some of the'
    >>> textacy.text_utils.keyword_in_context(doc.text, 'nation', window_width=35)
    ed States of America is an amazing  nation  that continues to lead the world t
    come the role model for developing  nation s attempting to give their people t
    ve before to better ourselves as a  nation , because what we change will set a
    nd education. Fortunately, we as a  nation  have the opportunity to fix the in
     sentences. Judges from across the  nation  have said for decades that they do
    reopened many racial wounds in our  nation . The war on drugs also put addicts

Extract various elements of interest from parsed documents:

.. code-block:: pycon

    >>> list(doc.ngrams(2, filter_stops=True, filter_punct=True, filter_nums=False))[:15]
    [Mr. President,
     Record copies,
     finalist essays,
     essays written,
     Vermont High,
     High School,
     School students,
     sixth annual,
     annual ``,
     essay contest,
     contest conducted,
     nearly 800,
     800 entries,
     material follows,
     United States]
    >>> list(doc.ngrams(3, filter_stops=True, filter_punct=True, min_freq=2))
    [lead the world,
     leading the world,
     2.2 million people,
     2.2 million people,
     mandatory minimum sentences,
     Mandatory minimum sentences,
     war on drugs,
     war on drugs]
    >>> list(doc.named_entities(drop_determiners=True, bad_ne_types='numeric'))
    [Record,
     Vermont High School,
     United States of America,
     Americans,
     U.S.,
     U.S.,
     African American]
    >>> pattern = textacy.regexes_etc.POS_REGEX_PATTERNS['en']['NP']
    >>> print(pattern)
    <DET>? <NUM>* (<ADJ> <PUNCT>? <CONJ>?)* (<NOUN>|<PROPN> <PART>?)+
    >>> list(doc.pos_regex_matches(pattern))[-10:]
    [experiment,
     many racial wounds,
     our nation,
     The war,
     drugs,
     addicts,
     bars,
     addiction,
     the problem,
     a mental health issue]
    >>> list(doc.semistructured_statements('it', cue='be'))
    [(it, is, important to humanize these statistics),
     (It, is, the third highest state expenditure, behind health care and education),
     (it, is, ; a mental health issue)]
    >>> doc.key_terms(algorithm='textrank', n=5)
    [('nation', 0.04315758994993049),
     ('world', 0.030590559641614556),
     ('incarceration', 0.029577233127175532),
     ('problem', 0.02411902162606202),
     ('people', 0.022631145896105508)]

Compute common statistical attributes of a text:

.. code-block:: pycon

    >>> doc.readability_stats
    {'automated_readability_index': 11.67580188679245,
     'coleman_liau_index': 10.89927271226415,
     'flesch_kincaid_grade_level': 10.711962264150948,
     'flesch_readability_ease': 56.022660377358505,
     'gunning_fog_index': 13.857358490566037,
     'n_chars': 2026,
     'n_polysyllable_words': 57,
     'n_sents': 20,
     'n_syllables': 648,
     'n_unique_words': 228,
     'n_words': 424,
     'smog_index': 12.773325707644965}

Count terms individually, and represent documents as a bag of terms with flexible weighting and inclusion criteria:

.. code-block:: pycon

    >>> doc.term_count('nation')
    6
    >>> bot = doc.as_bag_of_terms(weighting='tf', normalized=False, lemmatize='auto', ngram_range=(1, 1))
    >>> [(doc.spacy_stringstore[term_id], count)
    ...  for term_id, count in bot.most_common(n=10)]
    [('nation', 6),
     ('world', 4),
     ('incarceration', 4),
     ('people', 3),
     ('mandatory minimum', 3),
     ('lead', 3),
     ('minimum', 3),
     ('problem', 3),
     ('mandatory', 3),
     ('drug', 3)]


Project Links
-------------

- `textacy @ PyPi <https://pypi.python.org/pypi/textacy>`_
- `textacy @ GitHub <https://github.com/chartbeat-labs/textacy>`_
- `textacy @ ReadTheDocs <http://textacy.readthedocs.io/en/latest/>`_


Authors
-------

- Burton DeWilde (<burton@chartbeat.net>)


Unofficial Roadmap
------------------

- [x] import/export for common formats
- [x] serialization and streaming to/from disk
- [x] topic modeling via ``gensim`` and/or ``sklearn``
- [x] data viz for text analysis
- [ ] distributional representations (word2vec etc.) via either ``gensim`` or ``spacy``
- [ ] document similarity/clustering (?)
- [ ] basic dictionary-based methods e.g. sentiment analysis (?)
- [ ] text classification
- [ ] media frames analysis
