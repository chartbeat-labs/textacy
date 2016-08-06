# -*- coding: utf-8 -*-
"""
Load, process, iterate, transform, and save text content paired with metadata
— a document.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from collections import Counter
import os
import re
import warnings

from cytoolz import itertoolz
import spacy.about
from spacy.tokens.doc import Doc as sdoc
from spacy.tokens.token import Token as stoken
from spacy.tokens.span import Span as sspan

from textacy.compat import string_types, unicode_type
from textacy import data, fileio, spacy_utils, text_utils
from textacy.representations import network


class Document(object):
    """
    Pairing of the content of a document with its associated metadata, where the
    content has been tokenized, tagged, parsed, etc. by spaCy. Also keep references
    to the id-to-token mapping used by spaCy to efficiently represent the content.
    If initialized from plain text, this processing is performed automatically.

    ``Document`` also provides a convenient interface to information extraction,
    different doc representations, statistical measures of the content, and more.

    Args:
        text_or_sdoc (str or ``spacy.Doc``): text or spacy doc containing the
            content of this ``Document``; if str, it is automatically processed
            by spacy before assignment to the ``Document.spacy_doc`` attribute
        spacy_pipeline (``spacy.<lang>.<Lang>()``, optional): if a spacy pipeline
            has already been loaded or used to make the input ``spacy.Doc``, pass
            it in here to speed things up a touch; in general, only one of these
            pipelines should be loaded per process
        lang (str, optional): if doc's language is known, give its 2-letter code
            (https://cloud.google.com/translate/v2/using_rest#language-params);
            if None (default), lang will be automatically inferred from text
        metadata (dict, optional): dictionary of relevant information about the
            input ``text_or_sdoc``, e.g.::

                {'title': 'A Search for Second-generation Leptoquarks in pp Collisions at √s = 7 TeV with the ATLAS Detector',
                 'author': 'Burton DeWilde', 'pub_date': '2012-08-01'}
    """
    def __init__(self, text_or_sdoc, spacy_pipeline=None, lang=None, metadata=None):
        self.metadata = {} if metadata is None else metadata
        self._term_counts = Counter()

        if isinstance(text_or_sdoc, string_types):
            self.lang = text_utils.detect_language(text_or_sdoc) if not lang else lang
            if spacy_pipeline is None:
                spacy_pipeline = data.load_spacy(self.lang)
            # check for match between text and passed spacy_pipeline language
            else:
                if spacy_pipeline.lang != self.lang:
                    msg = 'Document.lang {} != spacy_pipeline.lang {}'.format(
                        self.lang, spacy_pipeline.lang)
                    raise ValueError(msg)
            self.spacy_vocab = spacy_pipeline.vocab
            self.spacy_stringstore = self.spacy_vocab.strings
            self.spacy_doc = spacy_pipeline(text_or_sdoc)

        elif isinstance(text_or_sdoc, sdoc):
            self.lang = spacy_pipeline.lang if spacy_pipeline is not None else \
                text_utils.detect_language(text_or_sdoc.text_with_ws)
            self.spacy_vocab = text_or_sdoc.vocab
            self.spacy_stringstore = self.spacy_vocab.strings
            self.spacy_doc = text_or_sdoc

        else:
            msg = 'Document must be initialized with {}, not {}'.format(
                {str, sdoc}, type(text_or_sdoc))
            raise ValueError(msg)

    def __repr__(self):
        snippet = self.text[:50].replace('\n', ' ')
        if len(snippet) == 50:
            snippet = snippet[:47] + '...'
        return 'Document({} tokens; "{}")'.format(len(self.spacy_doc), snippet)

    def __len__(self):
        return self.n_tokens

    def __getitem__(self, index):
        return self.spacy_doc[index]

    def __iter__(self):
        for tok in self.spacy_doc:
            yield tok

    ##########
    # FILEIO #

    def save(self, path, fname_prefix=None):
        """
        Save serialized Document content and metadata to disk.

        Args:
            path (str): directory on disk where content + metadata will be saved
            fname_prefix (str, optional): prepend standard filenames 'spacy_doc.bin'
                and 'metadata.json' with additional identifying information

        .. warn:: If the `spacy.Vocab` object used to save this document is not the
            same as the one used to load it, there will be problems! Consequently,
            this functionality is only useful as short-term but not long-term storage.
        """
        if fname_prefix:
            meta_fname = os.path.join(path, '_'.join([fname_prefix, 'metadata.json']))
            doc_fname = os.path.join(path, '_'.join([fname_prefix, 'spacy_doc.bin']))
        else:
            meta_fname = os.path.join(path, 'metadata.json')
            doc_fname = os.path.join(path, 'spacy_doc.bin')
        package_info = {'textacy_lang': self.lang,
                        'spacy_version': spacy.about.__version__}
        fileio.write_json(
            dict(package_info, **self.metadata), meta_fname)
        fileio.write_spacy_docs(self.spacy_doc, doc_fname)

    @classmethod
    def load(cls, path, fname_prefix=None):
        """
        Load serialized content and metadata from disk, and initialize a Document.

        Args:
            path (str): directory on disk where content + metadata are saved
            fname_prefix (str, optional): additional identifying information
                prepended to standard filenames 'spacy_doc.bin' and 'metadata.json'
                when saving to disk

        Returns:
            :class:`textacy.Document`

        .. warn:: If the `spacy.Vocab` object used to save this document is not the
            same as the one used to load it, there will be problems! Consequently,
            this functionality is only useful as short-term but not long-term storage.
        """
        if fname_prefix:
            meta_fname = os.path.join(path, '_'.join([fname_prefix, 'metadata.json']))
            docs_fname = os.path.join(path, '_'.join([fname_prefix, 'spacy_doc.bin']))
        else:
            meta_fname = os.path.join(path, 'metadata.json')
            docs_fname = os.path.join(path, 'spacy_doc.bin')
        metadata = list(fileio.read_json(meta_fname))[0]
        lang = metadata.pop('textacy_lang')
        spacy_version = metadata.pop('spacy_version')
        if spacy_version != spacy.about.__version__:
            msg = """
                the spaCy version used to save this Document to disk is not the
                same as the version currently installed ('{}' vs. '{}'); if the
                data underlying the associated `spacy.Vocab` has changed, this
                loaded Document may not be valid!
                """.format(spacy_version, spacy.about.__version__)
            warnings.warn(msg, UserWarning)
        spacy_vocab = data.load_spacy(lang).vocab
        return cls(list(fileio.read_spacy_docs(spacy_vocab, docs_fname))[0],
                   lang=lang, metadata=metadata)

    ####################
    # BASIC COMPONENTS #

    @property
    def tokens(self):
        """
        Yield the document's tokens, as tokenized by spaCy. Equivalent to
        iterating directly: `for token in document: <do stuff>`
        """
        for tok in self.spacy_doc:
            yield tok

    @property
    def sents(self):
        """Yield the document's sentences, as segmented by spaCy."""
        for sent in self.spacy_doc.sents:
            yield sent

    @property
    def n_tokens(self):
        """The number of tokens in the document -- including punctuation."""
        return len(self.spacy_doc)

    @property
    def n_sents(self):
        """The number of sentences in the document."""
        return sum(1 for _ in self.spacy_doc.sents)

    # @property
    # def n_words(self):
    #     """
    #     The number of words in the document -- i.e. the number of tokens,
    #     excluding punctuation.
    #     """
    #     return sum(1 for _ in self.words(filter_stops=False,
    #                                      filter_punct=True,
    #                                      filter_nums=False))

    # def n_paragraphs(self, pattern=r'\n\n+'):
    #     """The number of paragraphs in the document, as delimited by ``pattern``."""
    #     return sum(1 for _ in re.finditer(pattern, self.text)) + 1

    def merge(self, spans):
        """
        Merge spans *in-place* within doc so that each takes up a single token.
        Note: All cached methods on this doc will be cleared.

        Args:
            spans (iterable(``spacy.Span``)): for example, the results from
                :func:`extract.named_entities() <textacy.extract.named_entities>`
                or :func:`extract.pos_regex_matches() <textacy.extract.pos_regex_matches>`
        """
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
                        include_nes=False, include_ncs=False, include_kts=False):
        """
        Represent doc as a "bag of terms", an unordered set of (term id, term weight)
        pairs, where term weight may be by TF or TF*IDF.

        Args:
            weighting (str {'tf', 'tfidf'}, optional): weighting of term weights,
                either term frequency ('tf') or tf * inverse doc frequency ('tfidf')
            idf (dict, optional): if `weighting` = 'tfidf', idf's must be supplied
                externally, such as from a `Corpus` object
            lemmatize (bool or 'auto', optional): if True, lemmatize all terms
                when getting their frequencies
            ngram_range (tuple(int), optional): (min n, max n) values for n-grams
                to include in terms list; default (1, 1) only includes unigrams
            include_nes (bool, optional): if True, include named entities in terms list
            include_ncs (bool, optional): if True, include noun chunks in terms list
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
            include_ncs=include_ncs, include_kts=include_kts)

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

    def as_semantic_network(self, nodes='terms',
                            edge_weighting='default', window_width=10):
        """
        Represent doc as a semantic network, where nodes are either 'terms' or
        'sents', and edges between nodes may be weighted in different ways.

        Args:
            nodes (str, {'terms', 'sents'}): type of doc component to use as nodes
                in the semantic network
            edge_weighting (str, optional): type of weighting to apply to edges
                between nodes; if ``nodes == 'terms'``, options are {'cooc_freq', 'binary'},
                if ``nodes == 'sents'``, options are {'cosine', 'jaccard'}; if
                'default', 'cooc_freq' or 'cosine' will be automatically used
            window_width (int, optional): size of sliding window over terms that
                determines which are said to co-occur; only applicable if 'terms'

        Returns:
            :class:`networkx.Graph <networkx.Graph>`: where nodes represent either
                terms or sentences in doc; edges, the relationships between them

        .. seealso:: :func:`network.terms_to_semantic_network() <textacy.representations.network.terms_to_semantic_network>`
        .. seealso:: :func:`network.sents_to_semantic_network() <textacy.representations.network.sents_to_semantic_network>`
        """
        if nodes == 'terms':
            if edge_weighting == 'default':
                edge_weighting = 'cooc_freq'
            return network.terms_to_semantic_network(
                list(self.words()), window_width=window_width,
                edge_weighting=edge_weighting)
        elif nodes == 'sents':
            if edge_weighting == 'default':
                edge_weighting = 'cosine'
            return network.sents_to_semantic_network(
                list(self.sents), edge_weighting=edge_weighting)
        else:
            msg = 'nodes "{}" not valid; must be in {}'.format(nodes, {'terms', 'sents'})
            raise ValueError(msg)

    def as_terms_list(self, words=True, ngrams=(2, 3), named_entities=True,
                      dedupe=True, lemmatize=True, **kwargs):
        """
        Represent doc as a sequence of terms -- which aren't necessarily in order --
        including words (unigrams), ngrams (for a range of n), and named entities.
        NOTE: Despite the name, this is a generator function; to get a *list* of terms,
        just wrap the call like ``list(doc.as_terms_list())``.

        Args:
            words (bool, optional): if True (default), include words in the terms list
            ngrams (tuple(int), optional): include a range of ngrams in the terms list;
                default is ``(2, 3)``, i.e. bigrams and trigrams are included; if
                ngrams aren't wanted, set to False-y

                NOTE: if n=1 (words) is included here and ``words`` is True, n=1 is skipped
            named_entities (bool, optional): if True (default), include named entities
                in the terms list
            dedupe (bool, optional): if True (default), named entities are added first
                to the terms list, and any words or ngrams that exactly overlap with
                previously added entities are skipped to prevent double-counting;
                since words and ngrams (n > 1) are inherently exclusive, this only
                applies to entities; you almost certainly want this to be True
            lemmatize (bool, optional): if True (default), lemmatize all terms;
                otherwise, return the text as it appeared
            kwargs:
                filter_stops (bool)
                filter_punct (bool)
                filter_nums (bool)
                good_pos_tags (set(str))
                bad_pos_tags (set(str))
                min_freq (int)
                good_ne_types (set(str))
                bad_ne_types (set(str))
                drop_determiners (bool)

        Yields:
            str: the next term in the terms list
        """
        all_terms = []
        # special case: ensure that named entities aren't double-counted when
        # adding words or ngrams that were already added as named entities
        if dedupe is True and named_entities is True and (words is True or ngrams):
            ents = list(self.named_entities(**kwargs))
            ent_idxs = {(ent.start, ent.end) for ent in ents}
            all_terms.append(ents)
            if words is True:
                all_terms.append((word for word in self.words(**kwargs)
                                  if (word.idx, word.idx + 1) not in ent_idxs))
            if ngrams:
                for n in range(ngrams[0], ngrams[1] + 1):
                    if n == 1 and words is True:
                        continue
                    all_terms.append((ngram for ngram in self.ngrams(n, **kwargs)
                                      if (ngram.start, ngram.end) not in ent_idxs))
        # otherwise add everything in, duplicates and all
        else:
            if named_entities is True:
                all_terms.append(self.named_entities(**kwargs))
            if words is True:
                all_terms.append(self.words(**kwargs))
            if ngrams:
                for n in range(ngrams[0], ngrams[1] + 1):
                    if n == 1 and words is True:
                        continue
                    all_terms.append(self.ngrams(n, **kwargs))

        if lemmatize is True:
            for term in itertoolz.concat(all_terms):
                yield term.lemma_
        else:
            for term in itertoolz.concat(all_terms):
                yield term.text

    ##############
    # STATISTICS #

    def term_counts(self, lemmatize='auto', ngram_range=(1, 1),
                    include_nes=False, include_ncs=False, include_kts=False):
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
            include_ncs (bool, optional): if True, include noun chunks in terms list
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
        if include_ncs is True:
            self._term_counts = self._term_counts | Counter(
                get_id(nc) for nc in self.noun_chunks())
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
        if isinstance(term, unicode_type):
            term_text = term
            term_id = self.spacy_stringstore[term_text]
            term_len = term_text.count(' ') + 1
        elif isinstance(term, stoken):
            term_text = spacy_utils.normalized_str(term)
            term_id = self.spacy_stringstore[term_text]
            term_len = 1
        elif isinstance(term, sspan):
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
