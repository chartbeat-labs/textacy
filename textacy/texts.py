"""
Object-oriented interface for processing individual text documents as well as
collections (corpora). Wraps other modules' functionality with some amount of
caching, for efficiency.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import re

from cachetools import cachedmethod, LRUCache, hashkey
from collections import Counter
from functools import partial
from operator import attrgetter
from spacy.tokens.token import Token as spacy_token
from spacy.tokens.span import Span as spacy_span
from threading import RLock

from textacy import data, extract, spacy_utils, text_stats, text_utils, keyterms
from textacy.representations import vsm

LOCK = RLock()


class TextDoc(object):
    """
    Class that tokenizes, tags, and parses a text document, and provides an easy
    interface to information extraction, alternative document representations,
    and statistical measures of the text.

    Args:
        text (str)
        spacy_pipeline (``spacy.<lang>.<Lang>()``, optional)
        lang (str, optional)
        metadata (dict, optional)
        max_cachesize (int, optional)
    """
    def __init__(self, text, spacy_pipeline=None, lang='auto',
                 metadata=None, max_cachesize=5):
        self.metadata = {} if metadata is None else metadata
        self.lang = text_utils.detect_language(text) if lang == 'auto' else lang
        if spacy_pipeline is None:
            self.spacy_pipeline = data.load_spacy_pipeline(lang=self.lang)
        else:
            # check for match between text and supplied spacy pipeline language
            if spacy_pipeline.lang != self.lang:
                msg = 'TextDoc.lang {} != spacy_pipeline.lang {}'.format(
                    self.lang, spacy_pipeline.lang)
                raise ValueError(msg)
            else:
                self.spacy_pipeline = spacy_pipeline
        self.spacy_vocab = self.spacy_pipeline.vocab
        self.spacy_stringstore = self.spacy_vocab.strings
        self.spacy_doc = self.spacy_pipeline(text)
        self._term_counts = Counter()
        self._cache = LRUCache(maxsize=max_cachesize)

    def __repr__(self):
        return 'TextDoc({} tokens: {})'.format(
            len(self.spacy_doc), repr(self.text[:50].replace('\n',' ').strip() + '...'))

    def __len__(self):
        return self.n_tokens

    def __getitem__(self, index):
        return self.spacy_doc[index]

    def __iter__(self):
        for tok in self.spacy_doc:
            yield tok

    @property
    def tokens(self):
        """Yield the document's tokens as tokenized by spacy; same as ``__iter__``."""
        for tok in self.spacy_doc:
            yield tok

    @property
    def sents(self):
        """Yield the document's sentences as segmented by spacy."""
        for sent in self.spacy_doc.sents:
            yield sent

    def merge(self, spans):
        """
        Merge spans *in-place* within doc so that each takes up a single token.
        Note: All cached methods on this doc will be cleared.

        Args:
            spans (iterable(``spacy.Span``)): for example, the results from
                :func:`extract.named_entities() <textacy.extract.named_entities>`
                or :func:`extract.pos_regex_matches() <textacy.extract.pos_regex_matches>`
        """
        with LOCK:
            self._cache.clear()
        spacy_utils.merge_spans(spans)

    ###############
    # DOC AS TEXT #

    @property
    def text(self):
        """Return the document's raw text."""
        return self.spacy_doc.text_with_ws

    @property
    def tokenized_text(self):
        """Return text as an ordered, nested list of tokens per sentence."""
        return [[token.text for token in sent]
                for sent in self.spacy_doc.sents]

    @property
    def pos_tagged_text(self):
        """Return text as an ordered, nested list of (token, POS) pairs per sentence."""
        return [[(token.text, token.pos_) for token in sent]
                for sent in self.spacy_doc.sents]

    #######################
    # DOC REPRESENTATIONS #

    def as_bag_of_terms(self, weighting='tf', normalized=True, binary=False,
                        idf=None, lemmatize='auto',
                        ngram_range=(1, 1),
                        include_nes=False, include_nps=False, include_kts=False):
        """
        Represent doc as a "bag of terms", an unordered set of (term id, term weight)
        pairs, where term weight may be by TF or TF*IDF.

        Args:
            weighting (str {'tf', 'tfidf'}, optional): weighting of term weights,
                either term frequency ('tf') or tf * inverse doc frequency ('tfidf')
            idf (dict, optional): if `weighting` = 'tfidf', idf's must be supplied
                externally, such as from a `TextCorpus` object
            lemmatize (bool or 'auto', optional): if True, lemmatize all terms
                when getting their frequencies
            ngram_range (tuple(int), optional): (min n, max n) values for n-grams
                to include in terms list; default (1, 1) only includes unigrams
            include_nes (bool, optional): if True, include named entities in terms list
            include_nps (bool, optional): if True, include noun phrases in terms list
            include_kts (bool, optional): if True, include key terms in terms list
            normalized (bool, optional): if True, normalize term freqs by the
                total number of unique terms
            binary (bool optional): if True, set all (non-zero) term freqs equal to 1

        Returns:
            :class:`collections.Counter <collections.Counter>`: mapping of term ids
                to corresponding term weights
        """
        term_weights = self.term_counts(
            lemmatize=lemmatize, ngram_range=ngram_range, include_nes=include_nes,
            include_nps=include_nps, include_kts=include_kts)

        if binary is True:
            term_weights = Counter({key: 1 for key in term_weights.keys()})
        elif normalized is True:
            # n_terms = sum(term_freqs.values())
            n_tokens = self.n_tokens
            term_weights = Counter({key: val / n_tokens
                                    for key, val in term_weights.items()})

        if weighting == 'tfidf' and idf:
            term_weights = Counter({key: val * idf[key]
                                    for key, val in term_weights.items()})

        return term_weights

    def as_bag_of_concepts(self):
        raise NotImplementedError()

    def as_semantic_network(self):
        raise NotImplementedError()

    ##########################
    # INFORMATION EXTRACTION #

    @cachedmethod(attrgetter('_cache'), key=partial(hashkey, 'words'))
    def words(self, **kwargs):
        """
        Extract an ordered list of words from a spacy-parsed doc, optionally
        filtering words by part-of-speech (etc.) and frequency.

        .. seealso:: :func:`extract.words() <textacy.extract.words>` for all function kwargs.
        """
        return list(extract.words(self.spacy_doc, **kwargs))

    @cachedmethod(attrgetter('_cache'), key=partial(hashkey, 'ngrams'))
    def ngrams(self, n, **kwargs):
        """
        Extract an ordered list of n-grams (``n`` consecutive words) from doc,
        optionally filtering n-grams by the types and parts-of-speech of the
        constituent words.

        Args:
            n (int): number of tokens to include in n-grams;
                1 => unigrams, 2 => bigrams

        .. seealso:: :func:`extract.ngrams() <textacy.extract.ngrams>` for all function kwargs.
        """
        return list(extract.ngrams(self.spacy_doc, n, **kwargs))

    @cachedmethod(attrgetter('_cache'), key=partial(hashkey, 'named_entities'))
    def named_entities(self, **kwargs):
        """
        Extract an ordered list of named entities (PERSON, ORG, LOC, etc.) from
        doc, optionally filtering by the entity types and frequencies.

        .. seealso:: :func:`extract.named_entities() <textacy.extract.named_entities>`
        for all function kwargs.
        """
        return list(extract.named_entities(self.spacy_doc, **kwargs))

    @cachedmethod(attrgetter('_cache'), key=partial(hashkey, 'noun_chunks'))
    def noun_chunks(self, **kwargs):
        """
        Extract an ordered list of noun phrases from doc, optionally
        filtering by frequency and dropping leading determiners.

        .. seealso:: :func:`extract.noun_chunks() <textacy.extract.noun_chunks>`
        for all function kwargs.
        """
        return list(extract.noun_chunks(self.spacy_doc, **kwargs))

    @cachedmethod(attrgetter('_cache'), key=partial(hashkey, 'pos_regex_matches'))
    def pos_regex_matches(self, pattern):
        """
        Extract sequences of consecutive tokens from a spacy-parsed doc whose
        part-of-speech tags match the specified regex pattern.

        Args:
            pattern (str): Pattern of consecutive POS tags whose corresponding words
                are to be extracted, inspired by the regex patterns used in NLTK's
                ``nltk.chunk.regexp``. Tags are uppercase, from the universal tag set;
                delimited by < and >, which are basically converted to parentheses
                with spaces as needed to correctly extract matching word sequences;
                white space in the input doesn't matter.

                Examples (see :obj:`POS_REGEX_PATTERNS <textacy.regexes_etc.POS_REGEX_PATTERNS>`):

                * noun phrase: r'<DET>? (<NOUN>+ <ADP|CONJ>)* <NOUN>+'
                * compound nouns: r'<NOUN>+'
                * verb phrase: r'<VERB>?<ADV>*<VERB>+'
                * prepositional phrase: r'<PREP> <DET>? (<NOUN>+<ADP>)* <NOUN>+'
        """
        return list(extract.pos_regex_matches(self.spacy_doc, pattern))

    @cachedmethod(attrgetter('_cache'), key=partial(hashkey, 'subject_verb_object_triples'))
    def subject_verb_object_triples(self):
        """
        Extract an *un*ordered list of distinct subject-verb-object (SVO) triples
        from doc.
        """
        return list(extract.subject_verb_object_triples(self.spacy_doc))

    @cachedmethod(attrgetter('_cache'), key=partial(hashkey, 'acronyms_and_definitions'))
    def acronyms_and_definitions(self, **kwargs):
        """
        Extract a collection of acronyms and their most likely definitions,
        if available, from doc. If multiple definitions are found for a given acronym,
        only the most frequently occurring definition is returned.

        .. seealso:: :func:`extract.acronyms_and_definitions() <textacy.extract.acronyms_and_definitions>`
        for all function kwargs.
        """
        return extract.acronyms_and_definitions(self.spacy_doc, **kwargs)

    @cachedmethod(attrgetter('_cache'), key=partial(hashkey, 'semistructured_statements'))
    def semistructured_statements(self, entity, **kwargs):
        """
        Extract "semi-structured statements" from doc, each as a (entity, cue, fragment)
        triple. This is similar to subject-verb-object triples.

        Args:
            entity (str): a noun or noun phrase of some sort (e.g. "President Obama",
                "global warming", "Python")

        .. seealso:: :func:`extract.semistructured_statements() <textacy.extract.semistructured_statements>`
        for all function kwargs.
        """
        return list(extract.semistructured_statements(
            self.spacy_doc, entity, **kwargs))

    @cachedmethod(attrgetter('_cache'), key=partial(hashkey, 'direct_quotations'))
    def direct_quotations(self):
        """
        Baseline, not-great attempt at direction quotation extraction (no indirect
        or mixed quotations) using rules and patterns. English only.
        """
        return list(extract.direct_quotations(self.spacy_doc))

    @cachedmethod(attrgetter('_cache'), key=partial(hashkey, 'key_terms'))
    def key_terms(self, algorithm='sgrank', n=10):
        """
        Extract key terms from a document using `algorithm`.

        Args:
            algorithm (str {'sgrank', 'textrank', 'singlerank'}, optional): name
                of algorithm to use for key term extraction
            n (int or float, optional): if int, number of top-ranked terms to return
                as keyterms; if float, must be in the open interval (0.0, 1.0),
                representing the fraction of top-ranked terms to return as keyterms

        Raises:
            ValueError: if ``algorithm`` not in {'sgrank', 'textrank', 'singlerank'}
        """
        if algorithm == 'sgrank':
            return keyterms.sgrank(self.spacy_doc, window_width=1500, n_keyterms=n)
        elif algorithm == 'textrank':
            return keyterms.textrank(self.spacy_doc, n_keyterms=n)
        elif algorithm == 'singlerank':
            return keyterms.singlerank(self.spacy_doc, n_keyterms=n)
        else:
            raise ValueError('algorithm {} not a valid option'.format(algorithm))

    ##############
    # STATISTICS #

    @cachedmethod(attrgetter('_cache'), key=partial(hashkey, 'term_counts'))
    def term_counts(self, lemmatize='auto', ngram_range=(1, 1),
                    include_nes=False, include_nps=False, include_kts=False):
        """
        Get the number of occurrences ("counts") of each unique term in doc;
        terms may be words, n-grams, named entities, noun phrases, and key terms.

        Args:
            lemmatize (bool or 'auto', optional): if True, lemmatize all terms
                when getting their frequencies; if 'auto', lemmatize all terms
                that aren't proper nouns or acronyms
            ngram_range (tuple(int), optional): (min n, max n) values for n-grams
                to include in terms list; default (1, 1) only includes unigrams
            include_nes (bool, optional): if True, include named entities in terms list
            include_nps (bool, optional): if True, include noun phrases in terms list
            include_kts (bool, optional): if True, include key terms in terms list

        Returns:
            :class:`collections.Counter() <collections.Counter>`: mapping of unique
                term ids to corresponding term counts
        """
        if lemmatize == 'auto':
            get_id = lambda x: self.spacy_stringstore[spacy_utils.normalized_str(x)]
        elif lemmatize is True:
            get_id = lambda x: self.spacy_stringstore[x.lemma_]
        else:
            get_id = lambda x: self.spacy_stringstore[x.text]

        for n in range(ngram_range[0], ngram_range[1] + 1):
            if n == 1:
                self._term_counts = self._term_counts | Counter(
                    get_id(word) for word in self.words())
            else:
                self._term_counts = self._term_counts | Counter(
                    get_id(ngram) for ngram in self.ngrams(n))
        if include_nes is True:
            self._term_counts = self._term_counts | Counter(
                get_id(ne) for ne in self.named_entities())
        if include_nps is True:
            self._term_counts = self._term_counts | Counter(
                get_id(np) for np in self.noun_chunks())
        if include_kts is True:
            # HACK: key terms are currently returned as strings
            # TODO: cache key terms, and return them as spacy spans
            get_id = lambda x: self.spacy_stringstore[x]
            self._term_counts = self._term_counts | Counter(
                get_id(kt) for kt, _ in self.key_terms())

        return self._term_counts

    def term_count(self, term):
        """
        Get the number of occurrences ("count") of term in doc.

        Args:
            term (str or ``spacy.Token`` or ``spacy.Span``)

        Returns:
            int
        """
        # figure out what object we're dealing with here; convert as necessary
        if isinstance(term, str):
            term_text = term
            term_id = self.spacy_stringstore[term_text]
            term_len = term_text.count(' ') + 1
        elif isinstance(term, spacy_token):
            term_text = spacy_utils.normalized_str(term)
            term_id = self.spacy_stringstore[term_text]
            term_len = 1
        elif isinstance(term, spacy_span):
            term_text = spacy_utils.normalized_str(term)
            term_id = self.spacy_stringstore[term_text]
            term_len = len(term)

        term_count_ = self._term_counts[term_id]
        if term_count_ > 0:
            return term_count_
        # have we not already counted the appropriate `n` n-grams?
        if not any(self.spacy_stringstore[t].count(' ') == term_len
                   for t in self._term_counts):
            get_id = lambda x: self.spacy_stringstore[spacy_utils.normalized_str(x)]
            if term_len == 1:
                self._term_counts += Counter(get_id(w) for w in self.words())
            else:
                self._term_counts += Counter(get_id(ng) for ng in self.ngrams(term_len))
            term_count_ = self._term_counts[term_id]
            if term_count_ > 0:
                return term_count_
        # last resort: try a regular expression
        return sum(1 for _ in re.finditer(re.escape(term_text), self.text))

    @property
    def n_tokens(self):
        """The number of tokens in the document -- including punctuation."""
        return len(self.spacy_doc)

    def n_words(self, filter_stops=False, filter_punct=True, filter_nums=False):
        """
        The number of words in the document, with optional filtering of stop words,
        punctuation (on by default), and numbers.
        """
        return len(self.words(filter_stops=filter_stops,
                              filter_punct=filter_punct,
                              filter_nums=filter_nums))

    @property
    def n_sents(self):
        """The number of sentences in the document."""
        return sum(1 for _ in self.spacy_doc.sents)

    def n_paragraphs(self, pattern=r'\n\n+'):
        """The number of paragraphs in the document, as delimited by ``pattern``."""
        return sum(1 for _ in re.finditer(pattern, self.text)) + 1

    @cachedmethod(attrgetter('_cache'), key=partial(hashkey, 'readability_stats'))
    def readability_stats(self):
        return text_stats.readability_stats(self)


class TextCorpus(object):
    """
    A collection of :class:`TextDoc <textacy.texts.TextDoc>` s with some syntactic
    sugar and functions to compute corpus statistics.

    Initalize with a particular language.
    Add documents to corpus by :meth:`TextCorpus.add_text() <textacy.texts.TextCorpus.add_text>`.

    Iterate over corpus docs with ``for doc in TextCorpus``. Access individual docs
    by index (e.g. ``TextCorpus[0]`` or ``TextCorpus[0:10]``) or by boolean condition
    specified by lambda function (e.g. ``TextCorpus.get_docs(lambda x: len(x) > 100)``).
    """
    def __init__(self, lang):
        self.lang = lang
        self.spacy_pipeline = data.load_spacy_pipeline(lang=self.lang)
        self.spacy_vocab = self.spacy_pipeline.vocab
        self.spacy_stringstore = self.spacy_vocab.strings
        self.docs = []
        self.n_docs = 0
        self.n_tokens = 0

    def __repr__(self):
        return 'TextCorpus({} docs, {} tokens)'.format(self.n_docs, self.n_tokens)

    def __len__(self):
        return self.n_docs

    def __getitem__(self, index):
        return self.docs[index]

    def __iter__(self):
        for doc in self.docs:
            yield doc

    @classmethod
    def from_texts(cls, texts, lang):
        """
        Convenience function for creating a :class:`TextCorpus <textacy.texts.TextCorpus>`
        from an iterable of text strings. NOTE: Only useful for texts without additional metadata.

        Args:
            texts (iterable(str))
            lang (str)

        Returns:
            :class:`TextCorpus <textacy.texts.TextCorpus>`
        """
        textcorpus = cls(lang=lang)
        for text in texts:
            textcorpus.add_text(text, lang=lang)
        return textcorpus

    def add_text(self, text, lang='auto', metadata=None):
        """
        Create a :class:`TextDoc <textacy.texts.TextDoc>` from ``text`` and ``metadata``,
        then add it to the corpus.

        Args:
            text (str): raw text document to add to corpus as newly instantiated
                :class:`TextDoc <textacy.texts.TextDoc>`
            lang (str, optional):
            metadata (dict, optional): dictionary of document metadata, such as::

                {"title": "My Great Doc", "author": "Burton DeWilde"}

                NOTE: may be useful for retrieval via :func:`get_docs() <textacy.texts.TextCorpus.get_docs>`,
                e.g. ``TextCorpus.get_docs(lambda x: x.metadata["title"] == "My Great Doc")``
        """
        doc = TextDoc(text, spacy_pipeline=self.spacy_pipeline,
                      lang=lang, metadata=metadata)
        doc.corpus_index = self.n_docs
        doc.corpus = self
        self.docs.append(doc)
        self.n_docs += 1
        self.n_tokens += doc.n_tokens

    def add_doc(self, textdoc, print_warning=True):
        """
        Add an existing :class:`TextDoc <textacy.texts.TextDoc>` to the corpus as-is.
        NB: If ``textdoc`` is already added to this or another :class:`TextCorpus <textacy.texts.TextCorpus>`,
        a warning message will be printed and the ``corpus_index`` attribute will be
        overwritten, but you won't be prevented from adding the doc.

        Args:
            textdoc (:class:`TextDoc <textacy.texts.TextDoc>`)
            print_warning (bool, optional): if True, print a warning message if
                ``textdoc`` already added to a corpus; otherwise, don't ever print
                the warning and live dangerously
        """
        if textdoc.lang != self.lang:
            msg = 'TextDoc.lang {} != TextCorpus.lang {}'.format(textdoc.lang, self.lang)
            raise ValueError(msg)
        if hasattr(textdoc, 'corpus_index'):
            textdoc = copy.deepcopy(textdoc)
            if print_warning is True:
                print('**WARNING: TextDoc already associated with a TextCorpus; adding anyway...')
        textdoc.corpus_index = self.n_docs
        textdoc.corpus = self
        self.docs.append(textdoc)
        self.n_docs += 1
        self.n_tokens += textdoc.n_tokens

    def get_docs(self, match_condition, limit=None):
        """
        Iterate over all docs in corpus and return all (or N = ``limit``) for which
        ``match_condition(doc) is True``.

        Args:
            match_condition (func): function that operates on a :class:`TextDoc`
                and returns a boolean value; e.g. `lambda x: len(x) > 100` matches
                all docs with more than 100 tokens
            limit (int, optional): if not `None`, maximum number of matched docs
                to return

        Yields:
            :class:`TextDoc <textacy.texts.TextDoc>`: one per doc passing ``match_condition``
                up to ``limit`` docs
        """
        if limit is None:
            for doc in self:
                if match_condition(doc) is True:
                    yield doc
        else:
            n_matched_docs = 0
            for doc in self:
                if match_condition(doc) is True:
                    n_matched_docs += 1
                    if n_matched_docs > limit:
                        break
                    yield doc

    def remove_doc(self, index):
        """Remove the document at ``index`` from the corpus, and decrement the
        ``corpus_index`` attribute on all docs that come after it in the corpus."""
        for doc in self[index + 1:]:
            doc.corpus_index -= 1
        del self[index]

    def remove_docs(self, match_condition, limit=None):
        """
        Remove all (or N = ``limit``) docs in corpus for which ``match_condition(doc) is True``.
        Re-set all remaining docs' ``corpus_index`` attributes at the end.

        Args:
            match_condition (func): function that operates on a :class:`TextDoc <textacy.texts.TextDoc>`
                and returns a boolean value; e.g. ``lambda x: len(x) > 100`` matches
                all docs with more than 100 tokens
            limit (int, optional): if not None, maximum number of matched docs
                to remove
        """
        remove_indexes = [doc.corpus_index
                          for doc in self.get_docs(match_condition, limit=limit)]
        for index in remove_indexes:
            del self[index]
        # now let's re-set the `corpus_index` attribute for all docs at once
        for i, doc in enumerate(self):
            doc.corpus_index = i

    def to_term_doc_matrix(self, weighting='tf',
                           normalize=True, smooth_idf=True, sublinear_tf=False,
                           min_df=1, max_df=1.0, min_ic=0.0, max_n_terms=None,
                           ngram_range=(1, 1), include_nes=False,
                           include_nps=False, include_kts=False):
        """
        Transform corpus into a sparse CSR matrix, where each row i corresponds
        to a doc, each column j corresponds to a unique term, and matrix values
        (i, j) correspond to the tf or tf-idf weighting of term j in doc i.
        """
        return vsm.build_doc_term_matrix(
            self, self.spacy_vocab, weighting=weighting, normalize=normalize,
            smooth_idf=smooth_idf, sublinear_tf=sublinear_tf
             min_df=min_df, max_df=max_df, min_ic=min_ic,
            max_n_terms=max_n_terms)
