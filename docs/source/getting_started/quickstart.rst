Quickstart
==========

First things first: Import the package. Most functionality is available from
this top-level import, but we'll see that some features require their own imports.

.. code-block:: pycon

    >>> import textacy

Working with Text
-----------------

Let's start with a single text document:

.. code-block:: pycon

    >>> text = (
    ...     "Since the so-called \"statistical revolution\" in the late 1980s and mid 1990s, "
    ...     "much Natural Language Processing research has relied heavily on machine learning. "
    ...     "Formerly, many language-processing tasks typically involved the direct hand coding "
    ...     "of rules, which is not in general robust to natural language variation. "
    ...     "The machine-learning paradigm calls instead for using statistical inference "
    ...     "to automatically learn such rules through the analysis of large corpora "
    ...     "of typical real-world examples.")

**Note:** In almost all cases, textacy (as well as spaCy) expects to be
working with unicode text data. Throughout the code, this is indicated as ``str``
to be consistent with Python 3's default string type; users of Python 2, however,
must be mindful to use ``unicode``, and convert from the default (bytes) string
type as needed.

Before (or *in lieu of*) processing this text with spaCy, we can do a few things.
First, let's look for keywords-in-context, as a quick way to assess, by eye,
how a particular word or phrase is used in a body of text:

.. code-block:: pycon

    >>> textacy.text_utils.KWIC(text, "language", window_width=35)
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

Make a Doc
----------

Usually, though, we want to work with text that's been processed by spaCy:
tokenized, part-of-speech tagged, parsed, and so on. Since spaCy's pipelines
are language-dependent, we have to load a particular pipeline to match the text;
when working with texts from multiple languages, this can be a pain. Fortunately,
textacy includes automatic language detection to apply the right pipeline
to the text, and it caches the loaded language data to minimize wait time and
hassle. Making a ``Doc`` from text is easy:

.. code-block:: pycon

    >>> doc = textacy.make_spacy_doc(text)
    >>> doc._.preview
    'Doc(85 tokens: "Since the so-called "statistical revolution" in...")'

Under the hood, the text has been identified as English, and the default English-
language (``"en"``) pipeline has been loaded, cached, and applied to it. If you
need to customize the pipeline, you can still easily load and cache it, then
specify it yourself when initializing the doc:

.. code-block:: pycon

    >>> en = textacy.load_spacy_lang("en_core_web_sm", disable=("parser",))
    >>> doc = textacy.make_spacy_doc(text, lang=en)
    >>> doc._.preview
    'Doc(85 tokens: "Since the so-called "statistical revolution" in...")'

Oftentimes, text data comes paired with metadata, such as a title, author, or
publication date, and we'd like to keep them together. textacy makes this easy:

.. code-block:: pycon

    >>> metadata = {
    ...     "title": "Natural-language processing",
    ...     "url": "https://en.wikipedia.org/wiki/Natural-language_processing",
    ...     "source": "wikipedia",
    ... }
    >>> doc = textacy.make_spacy_doc((text, metadata))
    >>> doc._.meta["title"]
    'Natural-language processing'

textacy adds a variety of useful functionality to vanilla spaCy docs,
accessible via its ``._`` "underscore" property. For example: ``doc._.preview``
gives a convenient preview of the doc's contents, and ``doc._.meta`` returns
any metadata associated with the main text content.
See https://spacy.io/usage/processing-pipelines#custom-components-attributes
for implementation details.

**Note:** Older versions of textacy (<0.7.0) used a ``textacy.Doc`` class
as a convenient wrapper around an underlying spaCy ``Doc``, with additional
functionality available as class attributes and methods. Once spaCy started
natively supporting custom extensions on ``Doc`` objects (as well as custom
components in language processing pipelines), this approach made less sense.

Analyze a Doc
-------------

There are many ways to understand the content of a ``Doc``. For starters, let's
extract various elements of interest:

.. code-block:: pycon

    >>> list(textacy.extract.ngrams(
    ...     doc, 3, filter_stops=True, filter_punct=True, filter_nums=False))
    [1980s and mid,
     Natural Language Processing,
     Language Processing research,
     research has relied,
     heavily on machine,
     processing tasks typically,
     tasks typically involved,
     involved the direct,
     direct hand coding,
     coding of rules,
     robust to natural,
     natural language variation,
     learning paradigm calls,
     paradigm calls instead,
     inference to automatically,
     learn such rules,
     analysis of large,
     corpora of typical]
    >>> list(textacy.extract.ngrams(doc, 2, min_freq=2))
    [Natural Language, natural language]
    >>> list(textacy.extract.entities(doc, drop_determiners=True))
    [late 1980s and mid 1990s]
    >>> pattern = textacy.constants.POS_REGEX_PATTERNS["en"]["NP"]
    >>> pattern
    '<DET>? <NUM>* (<ADJ> <PUNCT>? <CONJ>?)* (<NOUN>|<PROPN> <PART>?)+'
    >>> list(textacy.extract.pos_regex_matches(doc, pattern))
    [statistical revolution,
     the late 1980s,
     mid 1990s,
     much Natural Language Processing research,
     machine learning,
     many language,
     tasks,
     the direct hand coding,
     rules,
     natural language variation,
     The machine,
     paradigm,
     statistical inference,
     such rules,
     the analysis,
     large corpora,
     typical real-world examples]

We can also identify key terms in a document by a number of algorithms:

.. code-block:: pycon

    >>> import textacy.keyterms  # note the import
    >>> textacy.keyterms.textrank(doc, normalize="lemma", n_keyterms=10)
    [('rule', 0.053388673426165274),
     ('language', 0.051393874215877614),
     ('machine', 0.05117898985460108),
     ('statistical', 0.04067363609786232),
     ('world', 0.03714841125591235),
     ('real', 0.034770679620166826),
     ('typical', 0.033277268136629116),
     ('corpora', 0.03214591053006896),
     ('1990', 0.03162874015505505),
     ('mid', 0.031567325030955926)]
    >>> textacy.keyterms.sgrank(doc, ngrams=(1, 2, 3, 4), normalize="lower", n_keyterms=0.1)
    [('natural language processing research', 0.31188112358833325),
     ('natural language variation', 0.09554941648195946),
     ('direct hand coding', 0.09461396545586934),
     ('mid 1990s', 0.05831079282180467),
     ('machine learning', 0.0552325339992006),
     ('late 1980s', 0.04713120721580818),
     ('general robust', 0.040647628278589344),
     ('statistical revolution', 0.03898147636679938)]

Or we can compute basic counts and various readability statistics:

.. code-block:: pycon

    >>> ts = textacy.TextStats(doc)
    >>> ts.n_unique_words
    57
    >>> ts.basic_counts
    {'n_sents': 3,
     'n_words': 73,
     'n_chars': 414,
     'n_syllables': 134,
     'n_unique_words': 57,
     'n_long_words': 30,
     'n_monosyllable_words': 38,
     'n_polysyllable_words': 19}
    >>> ts.flesch_kincaid_grade_level
    15.56027397260274
    >>> ts.readability_stats
    {'flesch_kincaid_grade_level': 15.56027397260274,
     'flesch_reading_ease': 26.84351598173518,
     'smog_index': 17.5058628484301,
     'gunning_fog_index': 20.144292237442922,
     'coleman_liau_index': 16.32928468493151,
     'automated_readability_index': 17.448173515981736,
     'lix': 65.42922374429223,
     'gulpease_index': 44.61643835616438,
     'wiener_sachtextformel': 11.857779908675797}

Lastly, we can transform a document into a "bag of terms", with flexible weighting
and term inclusion criteria:

.. code-block:: pycon

    >>> bot = doc._.to_bag_of_terms(
    ...     ngrams=(1, 2, 3), entities=True, weighting="count",
    ...     as_strings=True)
    >>> sorted(bot.items(), key=lambda x: x[1], reverse=True)[:15]
    [('call', 2),
     ('statistical', 2),
     ('machine', 2),
     ('language', 2),
     ('rule', 2),
     ('learn', 2),
     ('late 1980 and mid 1990', 1),
     ('revolution', 1),
     ('late', 1),
     ('1980', 1),
     ('mid', 1),
     ('1990', 1),
     ('Natural', 1),
     ('Language', 1),
     ('Processing', 1)]

Working with Many Texts
-----------------------

Many NLP tasks require datasets comprised of a large number of texts, which
are often stored on disk in one or multiple files. textacy makes it easy
to efficiently stream text and (text, metadata) pairs from disk, regardless of
the format or compression of the data.

Let's start with a single text file, where each line is a new text document::

    I love Daylight Savings Time: It's a biannual opportunity to find and fix obscure date-time bugs in your code. Can't wait for next time!
    Somewhere between "this is irritating but meh" and "blergh, why haven't I automated this yet?!" Fuzzy decision boundary.
    Spent an entire day translating structured data blobs into concise, readable sentences. Human language is hard.
    ...

In this case, the texts are tweets from my sporadic presence on Twitter ---
a fine example of small (and boring) data. Let's stream it from disk so we
can analyze it in textacy:

.. code-block:: pycon

    >>> texts = textacy.io.read_text('~/Desktop/burton-tweets.txt', lines=True)
    >>> for text in texts:
    ...     doc = textacy.make_spacy_doc(text)
    ...     print(doc._.preview)
    Doc(32 tokens; "I love Daylight Savings Time: It's a biannual o...")
    Doc(28 tokens; "Somewhere between "this is irritating but meh" ...")
    Doc(20 tokens; "Spent an entire day translating structured data...")
    ...

Okay, let's not *actually* analyze my ramblings on social media...

Instead, let's consider a more complicated dataset: a compressed JSON file in the
mostly-standard "lines" format, in which each line is a separate record with both
text data and metadata fields. As an example, we can use the "Capitol Words" dataset
integrated into textacy (see :ref:`ref-api-datasets` for details). The data
is downloadable from the `textacy-data GitHub repository
<https://github.com/bdewilde/textacy-data/releases/tag/capitol_words_py3_v1.0>`_.

.. code-block:: pycon

    >>> records = textacy.io.read_json(
    ...     "textacy/data/capitol_words/capitol-words-py3.json.gz",
    ...     mode="rt", lines=True)
    >>> for record in records:
    ...     doc = textacy.make_spacy_doc((record["text"], {"title": record["title"]}))
    ...     print(doc._.preview)
    ...     print("meta:", doc._.meta)
    ...     # do stuff...
    ...     break
    Doc(159 tokens; "Mr. Speaker, 480,000 Federal employees are work...")
    meta: {'title': 'JOIN THE SENATE AND PASS A CONTINUING RESOLUTION'}

For this and a few other datasets, convenient ``Dataset`` classes are already
implemented in textacy to help users get up and running, faster:

.. code-block:: pycon

    >>> import textacy.datasets  # note the import
    >>> ds = textacy.datasets.CapitolWords()
    >>> ds.download()
    >>> records = ds.records(speaker_name={"Hillary Clinton", "Barack Obama"})
    >>> next(records)
    ('I yield myself 15 minutes of the time controlled by the Democrats.',
     {'date': '2001-02-13',
      'congress': 107,
      'speaker_name': 'Hillary Clinton',
      'speaker_party': 'D',
      'title': 'MORNING BUSINESS',
      'chamber': 'Senate'})

Make a Corpus
-------------

A ``textacy.Corpus`` is an ordered collection of spaCy ``Doc`` s, all processed
by the same language pipeline. Let's continue with the Capitol Words dataset
and make a corpus from a stream of records. (**Note:** This may take a
few minutes.)

.. code-block:: pycon

    >>> corpus = textacy.Corpus("en", data=records)
    >>> corpus
    Corpus(1240 docs, 857548 tokens)

The language pipeline used to analyze documents in the corpus must be specified
on instantiation, but the data added to it may come in the form of one or a stream
of texts, records, or (valid) ``Doc`` s.

.. code-block:: pycon

    >>> textacy.Corpus(
    ...     textacy.load_spacy_lang("en_core_web_sm", disable=("parser", "tagger")),
    ...     data=ds.texts(speaker_party="R", chamber="House", limit=100))
    Corpus(100 docs, 31356 tokens)

You can use basic indexing as well as flexible boolean queries to select
documents in a corpus:

.. code-block:: pycon

    >>> corpus[-1]._.preview
    'Doc(2999 tokens: "In the Federalist Papers, we often hear the ref...")'
    >>> [doc._.preview for doc in corpus[10:15]]
    ['Doc(359 tokens: "My good friend from Connecticut raised an issue...")',
     'Doc(83 tokens: "My question would be: In response to the discus...")',
     'Doc(3338 tokens: "Madam President, I come to the floor today to s...")',
     'Doc(221 tokens: "Mr. President, I rise in support of Senator Tho...")',
     'Doc(3061 tokens: "Mr. President, I thank my distinguished colleag...")']
    >>> obama_docs = list(corpus.get(lambda doc: doc._.meta["speaker_name"] == "Barack Obama"))
    >>> len(obama_docs)
    411

It's important to note that all of the data in a ``textacy.Corpus`` is stored
in-memory, which makes a number of features much easier to implement.
Unfortunately, this means that the maximum size of a corpus will be bounded by RAM.

Analyze a Corpus
----------------

There are lots of ways to analyze the data in a corpus. Basic stats are
computed on the fly as documents are added (or removed) from a corpus:

.. code-block:: pycon

    >>> corpus.n_docs, corpus.n_sents, corpus.n_tokens
    (1240, 34530, 857548)

You can transform a corpus into a document-term matrix, with flexible tokenization,
weighting, and filtering of terms:

.. code-block:: pycon

    >>> import textacy.vsm  # note the import
    >>> vectorizer = textacy.Vectorizer(
    ...     tf_type="linear", apply_idf=True, idf_type="smooth", norm="l2",
    ...     min_df=2, max_df=0.95)
    >>> doc_term_matrix = vectorizer.fit_transform(
    ...     (doc._.to_terms_list(ngrams=1, entities=True, as_strings=True)
    ...      for doc in corpus))
    >>> print(repr(doc_term_matrix))
    <1240x12577 sparse matrix of type '<class 'numpy.float64'>'
    	with 217067 stored elements in Compressed Sparse Row format>

From a doc-term matrix, you can then train and interpret a topic model:

.. code-block:: pycon

    >>> import textacy.tm  # note the import
    >>> model = textacy.tm.TopicModel("nmf", n_topics=10)
    >>> model.fit(doc_term_matrix)
    >>> doc_topic_matrix = model.transform(doc_term_matrix)
    >>> doc_topic_matrix.shape
    (1240, 10)
    >>> for topic_idx, top_terms in model.top_topic_terms(vectorizer.id_to_term, top_n=10):
    ...     print("topic", topic_idx, ":", "   ".join(top_terms))
    topic 0 : New   people   child   work   need   York   bill   year   school   student
    topic 1 : rescind   quorum   order   unanimous   consent   ask   President   Mr.   Madam   objection
    topic 2 : dispense   reading   unanimous   consent   amendment   ask   President   Mr.   Madam   OFFICER
    topic 3 : motion   table   lay   reconsider   agree   thereto   Madam   preamble   intervene   print
    topic 4 : desire   Chamber   vote   Senators   rollcall   voter   amendment   2313   regular   cloture
    topic 5 : amendment   pende   aside   set   ask   unanimous   consent   Mr.   President   desk
    topic 6 : health   care   patient   Health   mental   quality   child   medical   information   coverage
    topic 7 : Iraq   war   troop   iraqi   Iraqis   policy   military   american   U.S.   force
    topic 8 : tax   budget   cut   debt   pay   deficit   $   fiscal   billion   spending
    topic 9 : Senator   Virginia   yield   West Virginia   West   question   thank   Massachusetts   objection   time

And that's just getting started! For now, though, I encourage you to pick a dataset
--- either your own or one already included in textacy --- and start exploring
the data. *Most* functionality is well-documented via in-code docstrings; to see
that information all together in nicely-formatted HTML, be sure to check out
the :ref:`ref-api-reference`.

Working with Many Languages
---------------------------

Since a ``Corpus`` uses the same spaCy language pipeline to process all input texts,
it only works in a mono-lingual context. In some cases, though, your collection
of texts may contain more than one language; for example, if I occasionally tweeted
in Spanish (sí, ¡se habla español!), the ``burton-tweets.txt`` dataset couldn't
be fed in its entirety into a single ``Corpus``. This is irritating, I know, but
there are some workarounds.

If you haven't already, download spaCy models for the languages you want to analyze ---
see :ref:`installation_downloading-data` for details. Then, if your use case
doesn't require ``Corpus`` functionality, you can iterate over the texts and
only analyze those for which models are available:

.. code-block:: pycon

    >>> for text in texts:
    ...     try:
    ...         doc = textacy.make_spacy_doc(text)
    ...     except OSError:
    ...         continue
    ...     # do stuff...

When the ``lang`` param is unspecified, textacy tries to auto-detect the text's
language and load the corresponding model; if that model is unavailable, spaCy
will raise an ``OSError``. This try/except also handles the case where
language detection fails and returns, say, "un" for "unknown".

If you do need a ``Corpus``, you can split the input texts by language into
distinct collections, then instantiate monolingual corpora on those collections.
For example:

.. code-block:: pycon

    >>> en_corpus = textacy.Corpus(
    ...     "en", data=(
    ...         text for text in texts
    ...         if textacy.lang_utils.detect_lang(text) == "en")
    ... )
    >>> es_corpus = textacy.Corpus(
    ...     "es", data=(
    ...         text for text in texts
    ...         if textacy.lang_utils.detect_lang(text) == "es")
    ... )

Both of these options are less convenient than I'd like, but hopefully they
get the job done.
