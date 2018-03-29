Basic Usage
===========

First things first: Import the package. Most functionality is available from
this top-level import, but we'll see that some features require their own imports.

.. code-block:: pycon

    >>> import textacy

Let's start with a single text document:

.. code-block:: pycon

    >>> text = (
    ...     'Since the so-called "statistical revolution" in the late 1980s and mid 1990s, '
    ...     'much Natural Language Processing research has relied heavily on machine learning. '
    ...     'Formerly, many language-processing tasks typically involved the direct hand coding '
    ...     'of rules, which is not in general robust to natural language variation. '
    ...     'The machine-learning paradigm calls instead for using statistical inference '
    ...     'to automatically learn such rules through the analysis of large corpora '
    ...     'of typical real-world examples.')

**Note:** In almost all cases, ``textacy`` (as well as ``spacy``) expects to be
working with unicode text data. Throughout the code, this is indicated as ``str``
to be consistent with Python 3's default string type; users of Python 2, however,
must be mindful to use ``unicode``, and convert from the default (bytes) string
type as needed.

Before (or *in lieu of*) processing this text with spaCy, we can do a few things.
First, let's look for keywords-in-context, as a quick way to assess, by eye,
how a particular word or phrase is used in a body of text:

.. code-block:: pycon

    >>> textacy.text_utils.KWIC(text, 'language', window_width=35)
     1980s and mid 1990s, much Natural  Language  Processing research has relied hea
    n machine learning. Formerly, many  language -processing tasks typically involve
    s not in general robust to natural  language  variation. The machine-learning pa

Sometimes, "raw" text is messy and must be cleaned up before analysis; other
times, an analysis simply benefits from well-standardized text. In either case,
the ``textacy.preprocessing`` module contains a number of functions to remove
URLs, punctuation, accents, HTML cruft, etc. as well as normalize whitespace.
For example:

.. code-block:: pycon

    >>> textacy.preprocess_text(text, lowercase=True, no_punct=True)
    'since the so called statistical revolution in the late 1980s and mid 1990s much natural language processing research has relied heavily on machine learning formerly many language processing tasks typically involved the direct hand coding of rules which is not in general robust to natural language variation the machine learning paradigm calls instead for using statistical inference to automatically learn such rules through the analysis of large corpora of typical real world examples'

Usually, though, we want to work with text that's been processed by spaCy:
tokenized, part-of-speech tagged, parsed, and so on. Since spaCy's pipelines
are language-dependent, we have to load a particular pipeline to match the text;
when working with texts from multiple languages, this can be a pain. Fortunately,
``textacy`` includes automatic language detection to apply the right pipeline
to the text, and it caches the loaded language data to minimize wait time and
hassle. Making a ``Doc`` from text is easy:

.. code-block:: pycon

    >>> doc = textacy.Doc(text)
    >>> doc
    Doc(85 tokens; "Since the so-called "statistical revolution" in...")

Under the hood, the text has been identified as English, and the default English-
language ("en") pipeline has been loaded, cached, and applied to it. If you need
to customize the pipeline, you can still easily load and cache it, then specify
it yourself when initializing the doc:

.. code-block:: pycon

    >>> en = textacy.load_spacy('en_core_web_sm', disable=('parser',))
    >>> textacy.Doc(text, lang=en)
    Doc(85 tokens; "Since the so-called "statistical revolution" in...")

TODO: Continue updating here.

Efficiently stream documents from disk and into a processed corpus:

.. code-block:: pycon

    >>> import textacy.datasets
    >>> cw = textacy.datasets.CapitolWords()
    >>> cw.download()
    >>> records = cw.records(speaker_name={'Hillary Clinton', 'Barack Obama'})
    >>> text_stream, metadata_stream = textacy.io.split_records(records, 'text')
    >>> corpus = textacy.Corpus('en', texts=text_stream, metadatas=metadata_stream)
    >>> corpus
    Corpus(1241 docs; 857058 tokens)

Represent corpus as a document-term matrix, with flexible weighting and filtering:

.. code-block:: pycon

    >>> vectorizer = textacy.Vectorizer(
    ...     tf_type='linear', apply_idf=True, idf_type='smooth', norm='l2',
    ...     min_df=2, max_df=0.95)
    >>> doc_term_matrix = vectorizer.fit_transform(
    ...     (doc.to_terms_list(ngrams=1, named_entities=True, as_strings=True)
    ...      for doc in corpus))
    >>> print(repr(doc_term_matrix))
    <1241x11708 sparse matrix of type '<class 'numpy.float64'>'
        with 215182 stored elements in Compressed Sparse Row format>

Train and interpret a topic model:

.. code-block:: pycon

    >>> model = textacy.TopicModel('nmf', n_topics=10)
    >>> model.fit(doc_term_matrix)
    >>> doc_topic_matrix = model.transform(doc_term_matrix)
    >>> doc_topic_matrix.shape
    (1241, 10)
    >>> for topic_idx, top_terms in model.top_topic_terms(vectorizer.id_to_term, top_n=10):
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
    >>> import textacy.keyterms
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

Compute basic counts and readability statistics for a given text:

.. code-block:: pycon

    >>> ts = textacy.TextStats(doc)
    >>> ts.n_unique_words
    1107
    >>> ts.basic_counts
    {'n_chars': 11498,
     'n_long_words': 512,
     'n_monosyllable_words': 1785,
     'n_polysyllable_words': 222,
     'n_sents': 99,
     'n_syllables': 3525,
     'n_unique_words': 1107,
     'n_words': 2516}
    >>> ts.flesch_kincaid_grade_level
    10.853709110179697
    >>> ts.readability_stats
    {'automated_readability_index': 12.801546064781363,
     'coleman_liau_index': 9.905629258346586,
     'flesch_kincaid_grade_level': 10.853709110179697,
     'flesch_readability_ease': 62.51222198133965,
     'gulpease_index': 55.10492845786963,
     'gunning_fog_index': 13.69506833036245,
     'lix': 45.76390294037353,
     'smog_index': 11.683781121521076,
     'wiener_sachtextformel': 5.401029023140788}

Count terms individually, and represent documents as a bag-of-terms with flexible
weighting and inclusion criteria:

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
