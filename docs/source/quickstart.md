# Quickstart

Install `textacy` and (if you haven't already) download a spaCy language pipeline for processing text:

```shell
$ pip install textacy
$ python -m spacy download en_core_web_sm
```

Make a spaCy ``Doc`` from text:

```pycon
>>> import textacy
>>> text = (
...     "Many years later, as he faced the firing squad, Colonel Aureliano Buendía "
...     "was to remember that distant afternoon when his father took him to discover ice. "
...     "At that time Macondo was a village of twenty adobe houses, built on the bank "
...     "of a river of clear water that ran along a bed of polished stones, which were "
...     "white and enormous, like prehistoric eggs. The world was so recent "
...     "that many things lacked names, and in order to indicate them it was necessary to point."
... )
>>> doc = textacy.make_spacy_doc(text, lang="en_core_web_sm")
>>> print(doc._.preview)
Doc(93 tokens: "Many years later, as he faced the firing squad,...")
```

Analyze the document:

```pycon
>>> from textacy import extract
>>> list(extract.entities(doc, include_types={"PERSON", "LOCATION"}))
[Aureliano Buendía, Macondo]
>>> list(extract.subject_verb_object_triples(doc))
[SVOTriple(subject=[he], verb=[faced], object=[firing, squad]),
 SVOTriple(subject=[father], verb=[took], object=[him]),
 SVOTriple(subject=[things], verb=[lacked], object=[names])]
>>> from textacy import text_stats as ts
>>> ts.n_words(doc), ts.n_unique_words(doc)
(84, 66)
>>> ts.diversity.ttr(doc)
0.7857142857142857
>>> ts.flesch_kincaid_grade_level(doc)
10.922857142857143
```

Make another document, and compare:

```pycon
>>> other_doc = textacy.make_spacy_doc(
...     "Finally, one Tuesday in December, at lunchtime, all at once he released the whole weight of his torment. "
...     "The children would remember for the rest of their lives the august solemnity with which their father, "
...     "devastated by his prolonged vigil and by the wrath of his imagination, revealed his discovery to them: "
...     "'The earth is round, like an orange.'",
...     lang="en_core_web_sm",
... )
>>> from textacy import similarity
>>> similarity.levenshtein(doc.text, other_doc.text)
0.2693965517241379
>>> similarity.cosine(
...     (tok.lemma_ for tok in extract.words(doc)),
...     (tok.lemma_ for tok in extract.words(other_doc))
... )
0.0914991421995628
>>> set(tok.text for tok in extract.words(doc)) & set(tok.text for tok in extract.words(other_doc))
{'father', 'like', 'remember'}
>>> ts.flesch_reading_ease(doc) > ts.flesch_reading_ease(other_doc)
True
```

Make many documents, with metadata:

```pycon
>>> records = [
...     (
...         "Many years later, as he faced the firing squad, Colonel Aureliano Buendía was to remember that distant afternoon when his father took him to discover ice. At that time Macondo was a village of twenty adobe houses, built on the bank of a river of clear water that ran along a bed of polished stones, which were white and enormous, like prehistoric eggs. The world was so recent that many things lacked names, and in order to indicate them it was necessary to point.",
...         {"title": "One Hundred Years of Solitude", "pub_yr": 1967},
...     ),
...     (
...         "Over the weekend the vultures got into the presidential palace by pecking through the screens on the balcony windows and the flapping of their wings stirred up the stagnant time inside, and at dawn on Monday the city awoke out of its lethargy of centuries with the warm, soft breeze of a great man dad and rotting grandeur.",
...         {"title": "The Autumn of the Patriarch", "pub_yr": 1975},
...     ),
...     (
...         "On the day they were going to kill him, Santiago Nasar got up at five-thirty in the morning to wait for the boat the bishop was coming on. He'd dreamed he was going through a grove of timber trees where a gentle drizzle was falling, and for an instant he was happy in his dream, but when he awoke he felt completely spattered with bird shit.",
...         {"title": "Chronicle of a Death Foretold", "pub_yr": 1981},
...     ),
...     (
...         "It was inevitable: the scent of bitter almonds always reminded him of the fate of unrequited love. Dr. Juvenal Urbino noticed it as soon as he entered the still darkened house where he had hurried on an urgent call to attend a case that for him had lost all urgency many years before. The Antillean refugee Jeremiah de Saint-Amour, disabled war veteran, photographer of children, and his most sympathetic opponent in chess, had escaped the torments of memory with the aromatic fumes of gold cyanide.",
...         {"title": "Love in the Time of Cholera", "pub_yr": 1985},
...     ),
...     (
...         "José Palacios, his oldest servant, found him floating naked with his eyes open in the purifying waters of his bath and thought he had drowned. He knew this was one of the many ways the General meditated, but the ecstasy in which he lay drifting seemed that of a man no longer of this world.",
...         {"title": "The General in His Labyrinth", "pub_yr": 1989},
...     ),
... ]
>>> corpus = textacy.Corpus("en_core_web_sm", records)
>>> print(corpus)
Corpus(5 docs, 383 tokens)
```

Analyze them:

```pycon
>>> corpus.n_sents
11
>>> import statistics
>>> corpus.agg_metadata("pub_yr", statistics.median)
1981
>>> sorted(corpus.word_counts(by="lemma_").items(), key=lambda x: x[1], reverse=True)[:15]
[('year', 2),
 ('time', 2),
 ('house', 2),
 ('water', 2),
 ('world', 2),
 ('go', 2),
 ('get', 2),
 ('dream', 2),
 ('awake', 2),
 ('man', 2),
 ('later', 1),
 ('face', 1),
 ('firing', 1),
 ('squad', 1),
 ('Colonel', 1)]
```

Transform them into other representations for further analysis:

```pycon
>>> from textacy.representations import Vectorizer
>>> vectorizer = Vectorizer(tf_type="linear", idf_type="smooth")
>>> doc_term_matrix = Vectorizer().fit_transform(
...     ((term.lemma_ for term in extract.terms(doc, ngs=1, ents=True)) for doc in corpus)
... )
>>> print(repr(doc_term_matrix))
<5x167 sparse matrix of type '<class 'numpy.int32'>'
    with 175 stored elements in Compressed Sparse Row format>
>>> doc_term_matrix[:, vectorizer.vocabulary_terms["year"]].toarray()
array([[1.69314718],
       [0.        ],
       [0.        ],
       [1.69314718],
       [0.        ]])
>>> from textacy.representations import build_cooccurrence_network
>>> cooc_graph = build_cooccurrence_network(
...     [[term.lemma_ for term in extract.terms(doc, ngs=1, ents=True)] for doc in corpus],
...     window_size=5,
... )
>>> cooc_graph.number_of_nodes(), cooc_graph.number_of_edges()
(167, 658)
>>> sorted(cooc_graph.adjacency())[1]
('Aureliano',
 {'Colonel': {'weight': 4},
  'face': {'weight': 1},
  'firing': {'weight': 2},
  'squad': {'weight': 3},
  'Buendía': {'weight': 4},
  'remember': {'weight': 3},
  'distant': {'weight': 2},
  'afternoon': {'weight': 1}})
```

Next steps:
- Go through `textacy`'s features in more detail and with more context in the Walkthrough.
- See example tasks worked end-to-end in the Tutorials.
- Consult the API Reference.
