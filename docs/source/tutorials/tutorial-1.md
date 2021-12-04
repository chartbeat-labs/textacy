# Context and Description of Workers in the U.S. Congress

In this tutorial, we will explore how certain members of the U.S. Congress have spoken about workers, based on a dataset of thousands of speeches sourced from the Congressional Record.

First, let's initialize and download the dataset, which comes built-in with `textacy`:

```pycon
>>> import textacy.datasets
>>> dataset = textacy.datasets.CapitolWords()
>>> dataset.info
{'name': 'capitol_words',
 'site_url': 'http://sunlightlabs.github.io/Capitol-Words/',
 'description': 'Collection of ~11k speeches in the Congressional Record given by notable U.S. politicians between Jan 1996 and Jun 2016.'}
>>> dataset.download()
```

Each record in this dataset contains the full text of and basic metadata about the speech. Let's take a peek at the first one, to get our bearings:

```pycon
>>> record = next(dataset.records(limit=1))
>>> record
Record(text='Mr. Speaker, 480,000 Federal employees are working without pay, a form of involuntary servitude; 280,000 Federal employees are not working, and they will be paid. Virtually all of these workers have mortgages to pay, children to feed, and financial obligations to meet.\nMr. Speaker, what is happening to these workers is immoral, is wrong, and must be rectified immediately. Newt Gingrich and the Republican leadership must not continue to hold the House and the American people hostage while they push their disastrous 7-year balanced budget plan. The gentleman from Georgia, Mr. Gingrich, and the Republican leadership must join Senator Dole and the entire Senate and pass a continuing resolution now, now to reopen Government.\nMr. Speaker, that is what the American people want, that is what they need, and that is what this body must do.', meta={'date': '1996-01-04', 'congress': 104, 'speaker_name': 'Bernie Sanders', 'speaker_party': 'I', 'title': 'JOIN THE SENATE AND PASS A CONTINUING RESOLUTION', 'chamber': 'House'})
```

This speech was delivered by Bernie Sanders back in 1996, when he was a member of the House of Representatives. By reading the text, we can see that it's about government workers during a shutdown — very relevant to our inquiry! :)

Considering the number of speeches, we'd like to avoid a full read-through and instead extract just the specific parts of interest. As a first step, let's use the `textacy.extract` subpackage to inspect our keywords in context.

```pycon
>>> import textacy.extract
>>> textacy.set_doc_extensions("extract")  # just setting these now -- we'll use them later!
>>> list(textacy.extract.keyword_in_context(record.text, "work(ing|ers?)", window_width=35))
[('ker, 480,000 Federal employees are ', 'working', ' without pay, a form of involuntary'),
 (' 280,000 Federal employees are not ', 'working', ', and they will be paid. Virtually '),
 ('ll be paid. Virtually all of these ', 'workers', ' have mortgages to pay, children to'),
 ('peaker, what is happening to these ', 'workers', ' is immoral, is wrong, and must be ')]
```

This is useful for developing our intuitions about how Bernie regards workers, but we'd prefer the information in a more structured form. Processing the text with spaCy will allow us to interrogate the text content in more sophisticated ways.

But first, we should preprocess the text to get rid of potential data quality issues (inconsistent quotation marks, whitespace, unicode characters, etc.) and other distractions that may affect our analysis. For example, maybe it would be better to replace all numbers with a constant placeholder value. For this, we'll use some of the functions available in `textacy.preprocessing`:

```pycon
>>> import textacy.preprocessing
>>> textacy.preprocessing.replace.numbers(record.text)
'Mr. Speaker, _NUMBER_ Federal employees are working without pay, a form of involuntary servitude; _NUMBER_ Federal employees are not working, and they will be paid. Virtually all of these workers have mortgages to pay, children to feed, and financial obligations to meet.\nMr. Speaker, what is happening to these workers is immoral, is wrong, and must be rectified immediately. Newt Gingrich and the Republican leadership must not continue to hold the House and the American people hostage while they push their disastrous _NUMBER_-year balanced budget plan. The gentleman from Georgia, Mr. Gingrich, and the Republican leadership must join Senator Dole and the entire Senate and pass a continuing resolution now, now to reopen Government.\nMr. Speaker, that is what the American people want, that is what they need, and that is what this body must do.'
```

Note that these changes are "destructive" — they've changed the data, and we can't reconstruct the original without keeping a copy around or re-loading it from disk. On second thought... let's leave the numbers alone.

However, we should still take care to normalize common text data errors. Let's combine multiple such preprocessors into a lightweight, callable pipeline that applies each sequentially:

```pycon
>>> preproc = textacy.preprocessing.make_pipeline(
...     textacy.preprocessing.normalize.unicode,
...     textacy.preprocessing.normalize.quotation_marks,
...     textacy.preprocessing.normalize.whitespace,
... )
>>> preproc_text = preproc(record.text)
>>> preproc_text[:200]
'Mr. Speaker, 480,000 Federal employees are working without pay, a form of involuntary servitude; 280,000 Federal employees are not working, and they will be paid. Virtually all of these workers have m'
```

To make a spaCy `Doc`, we need to apply a language-specific model pipeline to the text. (See the installation guide for details on how to download the necessary data!) Assuming most if not all of these speeches were given in English, let's use the ["en_core_web_sm"](https://spacy.io/models/en#en_core_web_sm) pipeline:

```pycon
>>> doc = textacy.make_spacy_doc((preproc_text, record.meta), lang="en_core_web_sm")
>>> doc._.preview
'Doc(161 tokens: "Mr. Speaker, 480,000 Federal employees are work...")'
>>> doc._.meta
{'date': '1996-01-04',
 'congress': 104,
 'speaker_name': 'Bernie Sanders',
 'speaker_party': 'I',
 'title': 'JOIN THE SENATE AND PASS A CONTINUING RESOLUTION',
 'chamber': 'House'}
```

Now, using the annotated part-of-speech tags, we can extract just the adjectives and determinants immediately preceding our keyword to get a sense of how workers are _described_:

```pycon
>>> patterns = [{"POS": {"IN": ["ADJ", "DET"]}, "OP": "+"}, {"ORTH": {"REGEX": "workers?"}}]
>>> list(textacy.extract.token_matches(doc, patterns))
[these workers, these workers]
```

Well, these particular examples aren't very interesting, but we'd definitely like to see the results aggregated over all speeches: _skilled_ workers, _American_ workers, _young_ workers, and so on.

To accomplish that, let's load many records into a `textacy.Corpus`. *Note:* For the sake of time, we'll limit ourselves to just the first 2000 — this can take a couple minutes!

```pycon
>>> records = dataset.records(limit=2000)
>>> preproc_records = ((preproc(text), meta) for text, meta in records)
>>> corpus = textacy.Corpus("en_core_web_sm", data=preproc_records)
>>> print(corpus)
Corpus(2000 docs, 1049192 tokens)
```

We can leverage the documents' metadata to get a better sense of what's in our corpus:

```pycon
>>> import collections
>>> corpus.agg_metadata("date", min), corpus.agg_metadata("date", max)
('1996-01-04', '1999-10-08')
>>> corpus.agg_metadata("speaker_name", collections.Counter)
Counter({'Bernie Sanders': 421,
         'Lindsey Graham': 98,
         'Rick Santorum': 533,
         'Joseph Biden': 691,
         'John Kasich': 257})
```

We see some familiar politicians, including current president Joe Biden and noted sycophant Lindsey Graham. Now that the documents are processed, let's extract matches from each, lemmatize their texts for consistency, and then inspect the most common descriptions of workers:

```pycon
>>> import itertools
>>> matches = itertools.chain.from_iterable(textacy.extract.token_matches(doc, patterns) for doc in corpus)
>>> collections.Counter(match.lemma_ for match in matches).most_common(20)
[('american worker', 95),
 ('average american worker', 21),
 ('the average american worker', 20),
 ('the worker', 15),
 ('social worker', 6),
 ('those worker', 5),
 ('a worker', 5),
 ('these worker', 4),
 ('young worker', 4),
 ('average worker', 4),
 ('an american worker', 4),
 ('the american worker', 4),
 ('federal worker', 3),
 ('that american worker', 3),
 ('that worker', 3),
 ('more worker', 3),
 ('nonunion worker', 3),
 ('the average worker', 3),
 ('young american worker', 2),
 ('every worker', 2)]
```

Apparently, these speakers had a preoccupation with American workers, average workers, and _average American_ workers. To better understand the context of these mentions, we can extract keyterms (the most important or "key" terms) from the documents in which they occured.

For example, here are the top 10 keyterms from that first Bernie speech in our dataset, extracted using a variation of the well-known TextRank algorithm:

```pycon
>>> corpus[0]._.extract_keyterms("textrank", normalize="lemma", window_size=10, edge_weighting="count", topn=10)
[('year balanced budget plan', 0.033721812470386026),
 ('Mr. Speaker', 0.032162715590532916),
 ('Mr. Gingrich', 0.031358819981176664),
 ('american people', 0.02612752273629427),
 ('republican leadership', 0.025418705021243045),
 ('federal employee', 0.021731159162187104),
 ('Newt Gingrich', 0.01988327361247088),
 ('pay', 0.018930131314143193),
 ('involuntary servitude', 0.015559235022115406),
 ('entire Senate', 0.015032623278646105)]
```

Now let's select the subset of speeches in which "worker(s)" were mentioned, extract the keyterms from each, then aggregate and rank the results.

```pycon
>>> kt_weights = collections.Counter()
>>> for doc in corpus.get(lambda doc: any(doc._.extract_regex_matches("workers?"))):
...     keyterms = doc._.extract_keyterms(
...         "textrank", normalize="lemma", window_size=10, edge_weighting="count", topn=10
...     )
...     kt_weights.update(dict(keyterms))
kt_weights.most_common(20)
[('average american worker', 0.2925480520167547),
 ('american worker', 0.21976899187473325),
 ('american people', 0.2131304787602286),
 ('real wage', 0.20937859927617333),
 ('Mr. Speaker', 0.19605562157627318),
 ('minimum wage today', 0.15268345523692883),
 ('young people', 0.13646481152944478),
 ('Social Security Social Security', 0.1361447369032916),
 ('Social Security Trust Fund', 0.12800826053880315),
 ('wage job', 0.1245701927182434),
 ('minimum wage', 0.1231061204217654),
 ('Mr. Chairman', 0.11731341389089317),
 ('low wage', 0.10747384130103463),
 ('time job', 0.10698519355007824),
 ('Multiple Chemical Sensitivity disorder', 0.09848493865271887),
 ('Mr. President', 0.09740781572099372),
 ('income people', 0.09569570041926843),
 ('Mr. Kucinich', 0.09241855965201626),
 ('violent crime trust fund', 0.08805244819537784),
 ('Social Security system', 0.08688954139546792)]
```

Perhaps unsurprisingly, "average american worker" ranks at the top of the list, but we can see from the rest of the list that they're brought up in discussion of jobs, the minimum wage, and Social Security. Makes sense!

In this tutorial, we've learned how to

- load text+metadata records from a dataset
- inspect and preprocess raw texts
- add a collection of documents processed by spaCy into a corpus
- inspect aggregated corpus metadata
- extract different kinds of structured data from one or many documents
