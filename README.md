## textacy: [tagline]

- elevator pitch
- key features outline
- installation instructions
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
