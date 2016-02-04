## textacy: [tagline]

- elevator pitch
- key features outline
- installation instructions / requirements
- quick start / complete usage example
- link(s) to full documentation

### Usage

For a single text:

```python
text = "Somewhere in la Mancha, in a place whose name I do not care to remember, a gentleman lived not long ago, one of those who has a lance and ancient shield on a shelf and keeps a skinny nag and a greyhound for racing."
metadata = {"title": "Don Quixote", "author": "Miguel de Cervantes"}
doc = TextDoc(text, metadata=metadata, lang="en")
```

For multiple texts:

```python
texts = ["Many years later, as he faced the firing squad, Colonel Aureliano Buendía was to remember that distant afternoon when his father took him to discover ice.",
         "The universe (which others call the Library) is composed of an indefinite and perhaps infinite number of hexagonal galleries, with vast air shafts between, surrounded by very low railings."]
corpus = TextCorpus.from_texts(texts, lang="en")
```

### Maintainers

- Burton DeWilde (<burton@chartbeat.net>)


### TODOs

- TODO: extract: return generators rather than lists?
- TODO: texts: figure out what to do when documents are modified in-place (`doc.merge`)
- TODO: texts: ^ related: when docs modified, erase cached_property attributes so they'll be re-caclulated
- TODO: texts: ^related: update doc merge functions when Honnibal updates API
- TODO: texts: what to do when new doc added to textcorpus does not have same language?
- TODO: texts: have textdocs inherit `_term_doc_freqs` from textcorpus?
- TODO: texts: add `doc_to_bag_of_terms()` func to transform?
- TODO: transform: condense csc matrix by mapping stringstore term ints to incremented vals, starting at 0
- TODO: drop scipy dependency and switch to honnibal's own sparse matrices
- TODO: preprocess: add basic tests for unidecode and ftfy functions
