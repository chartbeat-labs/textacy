========================================
textacy: higher-level NLP built on spaCy
========================================

``textacy`` is a Python library for performing higher-level natural language processing (NLP) tasks, built on the high-performance spaCy_ library. With the basics --- tokenization, part-of-speech tagging, dependency parsing, etc. --- offloaded to another library, ``textacy`` focuses on tasks facilitated by the ready availability of tokenized, POS-tagged, and parsed text.


Features
--------

- Stream text, json, csv, and spaCy binary data to and from disk
- Clean and normalize raw text, *before* analyzing it
- Explore included corpora of Congressional speeches and Supreme Court decisions, or stream documents from standard Wikipedia pages and Reddit comments datasets
- Access and filter basic linguistic elements, such as words and ngrams, noun chunks and sentences
- Extract named entities, acronyms and their definitions, direct quotations, key terms, and more from documents
- Compare strings, sets, and documents by a variety of similarity metrics
- Transform documents and corpora into vectorized and semantic network representations
- Train, interpret, visualize, and save ``sklearn``-style topic models using LSA, LDA, or NMF methods
- Identify a text's language, display key words in context (KWIC), true-case words, and navigate a parse tree

... and more!


Installation
------------

The simple way to install ``textacy`` is

.. code-block:: console

    $ pip install textacy

There are a couple optional dependencies, most notably ``matplotlib``, which is used for visualizations. If you need that functionality, be sure you have ``matplotlib`` installed already, or install it along with all of ``textacy`` 's other dependencies via

.. code-block:: console

    $ pip install textacy[viz]

If ``pip`` isn't an option, you can download and unzip the source ``tar.gz`` from  PyPi_, then

.. code-block:: console

    $ python setup.py install


Example
-------

.. code-block:: pycon

    >>> import textacy

Efficiently stream documents from disk and into a processed corpus:

.. code-block:: pycon

    >>> cw = textacy.corpora.CapitolWords()
    >>> docs = cw.records(speaker_name={'Hillary Clinton', 'Barack Obama'})
    >>> content_stream, metadata_stream = textacy.fileio.split_record_fields(
    ...     docs, 'text')
    >>> corpus = textacy.Corpus('en', texts=content_stream, metadatas=metadata_stream)
    >>> corpus
    Corpus(1241 docs; 857058 tokens)

Represent corpus as a document-term matrix, with flexible weighting and filtering:

.. code-block:: pycon

    >>> doc_term_matrix, id2term = textacy.vsm.doc_term_matrix(
    ...     (doc.to_terms_list(ngrams=1, named_entities=True, as_strings=True)
    ...      for doc in corpus),
    ...     weighting='tfidf', normalize=True, smooth_idf=True, min_df=2, max_df=0.95)
    >>> print(repr(doc_term_matrix))
    <1241x11364 sparse matrix of type '<class 'numpy.float64'>'
	   with 211602 stored elements in Compressed Sparse Row format>

Train and interpret a topic model:

.. code-block:: pycon

    >>> model = textacy.tm.TopicModel('nmf', n_topics=10)
    >>> model.fit(doc_term_matrix)
    >>> doc_topic_matrix = model.transform(doc_term_matrix)
    >>> doc_topic_matrix.shape
    (1241, 10)
    >>> for topic_idx, top_terms in model.top_topic_terms(id2term, top_n=10):
    ...     print('topic', topic_idx, ':', '   '.join(top_terms))
    topic 0 : new   people   's   american   senate   need   iraq   york   americans   work
    topic 1 : rescind   quorum   order   consent   unanimous   ask   president   mr.   madam   aside
    topic 2 : dispense   reading   amendment   unanimous   consent   ask   president   mr.   pending   aside
    topic 3 : health   care   child   mental   quality   patient   medical   program   information   family
    topic 4 : student   school   education   college   child   teacher   high   program   loan   year
    topic 5 : senators   desiring   chamber   vote   4,600   amtrak   rail   airline   litigation   expedited
    topic 6 : senate   thursday   wednesday   session   unanimous   consent   authorize   p.m.   committee   ask
    topic 7 : medicare   drug   senior   medicaid   prescription   benefit   plan   cut   cost   fda
    topic 8 : flu   vaccine   avian   pandemic   roberts   influenza   seasonal   outbreak   health   cdc
    topic 9 : virginia   west virginia   west   senator   yield   question   thank   objection   inquiry   massachusetts

Basic indexing as well as flexible selection of documents in a corpus:

.. code-block:: pycon

    >>> obama_docs = list(corpus.get(
    ...     lambda doc: doc.metadata['speaker_name'] == 'Barack Obama'))
    >>> len(obama_docs)
    411
    >>> doc = corpus[-1]
    >>> doc
    Doc(2999 tokens; "In the Federalist Papers, we often hear the ref...")

Preprocess plain text, or highlight particular terms in it:

.. code-block:: pycon

    >>> textacy.preprocess_text(doc.text, lowercase=True, no_punct=True)[:70]
    'in the federalist papers we often hear the reference to the senates ro'
    >>> textacy.text_utils.keyword_in_context(doc.text, 'America', window_width=35)
    g on this tiny piece of Senate and  America n history. Some 10 years ago, I ask
    o do the hard work in New York and  America , who get up every day and do the v
    say: You know, you never can count  America  out. Whenever the chips are down,
     what we know will give our fellow  America ns a better shot at the kind of fut
    aith in this body and in my fellow  America ns. I remain an optimist, that Amer
    ricans. I remain an optimist, that  America 's best days are still ahead of us.

Extract various elements of interest from parsed documents:

.. code-block:: pycon

    >>> list(textacy.extract.ngrams(
    ...     doc, 2, filter_stops=True, filter_punct=True, filter_nums=False))[:15]
    [Federalist Papers,
     Senate's,
     's role,
     violent passions,
     pernicious resolutions,
     everlasting credit,
     common ground,
     8 years,
     tiny piece,
     American history,
     10 years,
     years ago,
     New York,
     fellow New,
     New Yorkers]
    >>> list(textacy.extract.ngrams(
    ...     doc, 3, filter_stops=True, filter_punct=True, min_freq=2))
    [fellow New Yorkers,
     World Trade Center,
     Senator from New,
     World Trade Center,
     Senator from New,
     lot of fun,
     fellow New Yorkers,
     lot of fun]
    >>> list(textacy.extract.named_entities(
    ...     doc, drop_determiners=True, exclude_types='numeric'))[:10]
    [Senate,
     Senate,
     American,
     New York,
     New Yorkers,
     Senate,
     Barbara Mikulski,
     Senate,
     Pennsylvania Avenue,
     Senate]
    >>> pattern = textacy.constants.POS_REGEX_PATTERNS['en']['NP']
    >>> pattern
    <DET>? <NUM>* (<ADJ> <PUNCT>? <CONJ>?)* (<NOUN>|<PROPN> <PART>?)+
    >>> list(textacy.extract.pos_regex_matches(doc, pattern))[:10]
    [the Federalist Papers,
     the reference,
     the Senate's role,
     the consequences,
     sudden and violent passions,
     intemperate and pernicious resolutions,
     the everlasting credit,
     wisdom,
     our Founders,
     an effort]
    >>> list(textacy.extract.semistructured_statements(doc, 'I', cue='be'))
    [(I, was, on the other end of Pennsylvania Avenue),
     (I, was, , a very new Senator, and my city and my State had been devastated),
     (I, am, grateful to have had Senator Schumer as my partner and my ally),
     (I, am, very excited about what can happen in the next 4 years),
     (I, been, a New Yorker, but I know I always will be one)]
    >>> textacy.keyterms.textrank(doc, n_keyterms=10)
    [('day', 0.01608508275877894),
     ('people', 0.015079868730811194),
     ('year', 0.012330783590843065),
     ('way', 0.011732786337383587),
     ('colleague', 0.010794482493897155),
     ('new', 0.0104941198408241),
     ('time', 0.010016582029543003),
     ('work', 0.0096498231660789),
     ('lot', 0.008960478625039818),
     ('great', 0.008552318032915361)]

Compute common statistical attributes of a text:

.. code-block:: pycon

    >>> textacy.text_stats.readability_stats(doc)
    {'automated_readability_index': 12.549920902265107,
     'coleman_liau_index': 9.882109957869638,
     'flesch_kincaid_grade_level': 10.65744148341702,
     'flesch_readability_ease': 63.02302106124765,
     'gunning_fog_index': 13.493768200349448,
     'n_chars': 11498,
     'n_polysyllable_words': 222,
     'n_sents': 101,
     'n_syllables': 3525,
     'n_unique_words': 1107,
     'n_words': 2516,
     'smog_index': 11.598657798783282}

Count terms individually, and represent documents as a bag-of-terms with flexible weighting and inclusion criteria:

.. code-block:: pycon

    >>> doc.count('America')
    3
    >>> bot = doc.to_bag_of_terms(ngrams={2, 3}, as_strings=True)
    >>> sorted(bot.items(), key=lambda x: x[1], reverse=True)[:10]
    [('new york', 18),
     ('senate', 8),
     ('first', 6),
     ('state', 4),
     ('9/11', 3),
     ('look forward', 3),
     ('america', 3),
     ('new yorkers', 3),
     ('chuck', 3),
     ('lot of fun', 2)]


Project Links
-------------

- `textacy @ PyPi <https://pypi.python.org/pypi/textacy>`_
- `textacy @ GitHub <https://github.com/chartbeat-labs/textacy>`_
- `textacy @ ReadTheDocs <http://textacy.readthedocs.io/en/latest/>`_


Authors
-------

- Burton DeWilde (<burton@chartbeat.net>)


Roadmap
-------

#. document clustering
#. media framing analysis (?)
#. deep neural network model for text summarization
#. deep neural network model for sentiment analysis


.. _spaCy: https://spacy.io/
.. _PyPi: https://pypi.python.org/pypi/textacy
