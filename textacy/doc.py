# -*- coding: utf-8 -*-
"""
Load, process, iterate, transform, and save text content paired with metadata
— a document.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from collections import Counter
import os
import warnings

from cytoolz import itertoolz
import spacy.about
from spacy import attrs
from spacy.tokens.doc import Doc as SpacyDoc
from spacy.tokens.span import Span as SpacySpan
from spacy.tokens.token import Token as SpacyToken

import textacy
from textacy.compat import string_types, unicode_type
from textacy import data, fileio, spacy_utils, text_utils
from textacy.representations import network


class Doc(object):
    """
    Pairing of the content of a document with its associated metadata, where the
    content has been tokenized, tagged, parsed, etc. by spaCy. Also keep references
    to the id-to-token mapping used by spaCy to efficiently represent the content.
    If initialized from plain text, this processing is performed automatically.

    ``Doc`` also provides a convenient interface to information extraction,
    different doc representations, statistical measures of the content, and more.

    Args:
        text_or_sdoc (str or ``spacy.Doc``): text or spacy doc containing the
            content of this ``Doc``; if str, it is automatically processed
            by spacy before assignment to the ``Doc.spacy_doc`` attribute
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
        self.metadata = metadata or {}
        self._term_counts = Counter()

        if isinstance(text_or_sdoc, unicode_type):
            self.lang = lang or text_utils.detect_language(text_or_sdoc)
            if spacy_pipeline is None:
                spacy_pipeline = data.load_spacy(self.lang)
            # check for match between text and passed spacy_pipeline language
            else:
                if spacy_pipeline.lang != self.lang:
                    msg = 'Doc.lang {} != spacy_pipeline.lang {}'.format(
                        self.lang, spacy_pipeline.lang)
                    raise ValueError(msg)
            self.spacy_vocab = spacy_pipeline.vocab
            self.spacy_stringstore = self.spacy_vocab.strings
            self.spacy_doc = spacy_pipeline(text_or_sdoc)

        elif isinstance(text_or_sdoc, SpacyDoc):
            if spacy_pipeline is not None:
                self.lang = spacy_pipeline.lang
            else:
                self.lang = text_utils.detect_language(text_or_sdoc.text_with_ws)
            self.spacy_vocab = text_or_sdoc.vocab
            self.spacy_stringstore = self.spacy_vocab.strings
            self.spacy_doc = text_or_sdoc

        else:
            msg = 'Doc must be initialized with {}, not {}'.format(
                {str, SpacyDoc}, type(text_or_sdoc))
            raise ValueError(msg)

    def __repr__(self):
        snippet = self.text[:50].replace('\n', ' ')
        if len(snippet) == 50:
            snippet = snippet[:47] + '...'
        return 'Doc({} tokens; "{}")'.format(len(self.spacy_doc), snippet)

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
        Save serialized Doc content and metadata to disk.

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
        Load serialized content and metadata from disk, and initialize a Doc.

        Args:
            path (str): directory on disk where content + metadata are saved
            fname_prefix (str, optional): additional identifying information
                prepended to standard filenames 'spacy_doc.bin' and 'metadata.json'
                when saving to disk

        Returns:
            :class:`textacy.Doc`

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
                the spaCy version used to save this Doc to disk is not the
                same as the version currently installed ('{}' vs. '{}'); if the
                data underlying the associated `spacy.Vocab` has changed, this
                loaded Doc may not be valid!
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
        # reset counts, since merging spans invalidates existing counts
        self._counts = Counter()
        self._counted_ngrams = set()

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

    #################
    # TRANSFORM DOC #

    def to_terms_list(self, ngrams=(1, 2, 3), named_entities=True,
                      lemmatize=True, lowercase=False, as_strings=False,
                      **kwargs):
        """
        Transform ``Doc`` into a sequence of ngrams and/or named entities, which
        aren't necessarily in order of appearance.

        Args:
            ngrams (int or Set[int]): n of which n-grams to include; ``(1, 2, 3)``
                (default) includes unigrams (words), bigrams, and trigrams; `2`
                if only bigrams are wanted; falsy (e.g. False) to not include any
            named_entities (bool): if True (default), include named entities
                in the terms list; note: if ngrams are also included, named
                entities are added *first*, and any ngrams that exactly overlap
                with an entity are skipped to prevent double-counting
            lemmatize (bool): if True (default), lemmatize all terms
            lowercase (bool): if True and `lemmatize` is False, words are lower-
                cased
            as_strings (bool): if True, terms are returned as strings; if False
                (default), terms are returned as their unique integer ids
            kwargs:
                - filter_stops (bool)
                - filter_punct (bool)
                - filter_nums (bool)
                - include_pos (str or Set[str])
                - exclude_pos (str or Set[str])
                - min_freq (int)
                - include_types (str or Set[str])
                - exclude_types (str or Set[str]
                - drop_determiners (bool)
                see :func:`extract.words <textacy.extract.words>`,
                :func:`extract.ngrams <textacy.extract.ngrams>`,
                and :func:`extract.named_entities <textacy.extract.named_entities>`
                for more information on these parameters

        Yields:
            int or str: the next term in the terms list, either as a unique
                integer id or as a string

        .. note:: Despite the name, this is a generator function; to get a `list`
            of terms, call ``list(doc.to_terms_list())``.
        """
        if not named_entities and not ngrams:
            raise ValueError()
        if isinstance(ngrams, int):
            ngrams = (ngrams,)

        terms = []
        # special case: ensure that named entities aren't double-counted when
        # adding words or ngrams that were already added as named entities
        if named_entities is True and ngrams:
            ents = tuple(textacy.extract.named_entities(self, **kwargs))
            ent_idxs = {(ent.start, ent.end) for ent in ents}
            terms.append(ents)
            for n in ngrams:
                if n == 1:
                    terms.append((word for word in textacy.extract.words(self, **kwargs)
                                  if (word.idx, word.idx + 1) not in ent_idxs))
                else:
                    terms.append((ngram for ngram in textacy.extract.ngrams(self, n, **kwargs)
                                  if (ngram.start, ngram.end) not in ent_idxs))
        # otherwise, no need to check for overlaps
        else:
            if named_entities is True:
                terms.append(textacy.extract.named_entities(self, **kwargs))
            else:
                for n in ngrams:
                    if n == 1:
                        terms.append(textacy.extract.words(self, **kwargs))
                    else:
                        terms.append(textacy.extract.ngrams(self, n, **kwargs))

        terms = itertoolz.concat(terms)

        # convert token and span objects into integer ids
        if as_strings is False:
            if lemmatize is True:
                for term in terms:
                    try:
                        yield term.lemma
                    except AttributeError:
                        yield self.spacy_stringstore[term.lemma_]
            elif lowercase is True:
                for term in terms:
                    try:
                        yield term.lower
                    except AttributeError:
                        yield self.spacy_stringstore[term.orth_.lower()]
            else:
                for term in terms:
                    try:
                        yield term.orth
                    except AttributeError:
                        yield self.spacy_stringstore[term.orth_]
        # convert token and span objects into strings
        else:
            if lemmatize is True:
                for term in terms:
                    yield term.lemma_
            elif lowercase is True:
                for term in terms:
                    try:
                        yield term.lower_
                    except AttributeError:
                        yield term.orth_.lower()
            else:
                for term in terms:
                    yield term.orth_

    def to_bag_of_words(self, lemmatize=True, lowercase=False, normalize=False,
                        as_strings=False):
        """
        Transform ``Doc`` into a bag-of-words: the set of unique words in ``Doc``
        mapped to their frequency of occurrence.

        Args:
            lemmatize (bool): if True, words are lemmatized before counting;
                for example, 'happy', 'happier', and 'happiest' would be grouped
                together as 'happy', with a count of 3
            lowercase (bool): if True and `lemmatize` is False, words are lower-
                cased before counting; for example, 'happy' and 'Happy' would be
                grouped together as 'happy', with a count of 2
            normalize (bool): if True, normalize individual words' counts by the
                total token count, giving instead their *relative* frequency of
                occurrence in ``Doc``; note: the resulting set of values won't
                (necessarily) sum to 1.0, since punctuation and stop words are
                filtered out after counts are normalized
            as_strings (bool): if True, words are returned as strings; if False
                (default), words are returned as their unique integer ids

        Returns:
            dict: mapping of a unique word id or string (depending on the value
                of `as_strings`) to its absolute or relative frequency of
                occurrence (depending on the value of `normalize`)
        """
        count_by = (attrs.LEMMA if lemmatize is True else
                    attrs.LOWER if lowercase is True else attrs.ORTH)
        word_to_weight = self.spacy_doc.count_by(count_by)
        if normalize is True:
            n_tokens = self.n_tokens
            word_to_weight = {id_: weight / n_tokens
                              for id_, weight in word_to_weight.items()}
        bow = {}
        if as_strings is False:
            for id_, count in word_to_weight.items():
                lexeme = self.spacy_vocab[id_]
                if lexeme.is_stop or lexeme.is_punct or lexeme.is_space:
                    continue
                bow[id_] = count
        else:
            for id_, count in word_to_weight.items():
                lexeme = self.spacy_vocab[id_]
                if lexeme.is_stop or lexeme.is_punct or lexeme.is_space:
                    continue
                bow[self.spacy_stringstore[id_]] = count
        return bow

    def to_bag_of_terms(self, ngrams=(1, 2, 3), named_entities=True,
                        lemmatize=True, lowercase=False,
                        normalize=False, as_strings=False, **kwargs):
        """
        Transform ``Doc`` into a bag-of-terms: the set of unique terms in ``Doc``
        mapped to their frequency of occurrence, where "terms" includes ngrams
        and/or named entities.

        Args:
            ngrams (int or Set[int]): n of which n-grams to include; ``(1, 2, 3)``
                (default) includes unigrams (words), bigrams, and trigrams; `2`
                if only bigrams are wanted; falsy (e.g. False) to not include any
            named_entities (bool): if True (default), include named entities;
                note: if ngrams are also included, any ngrams that exactly
                overlap with an entity are skipped to prevent double-counting
            lemmatize (bool): if True, words are lemmatized before counting;
                for example, 'happy', 'happier', and 'happiest' would be grouped
                together as 'happy', with a count of 3
            lowercase (bool): if True and `lemmatize` is False, words are lower-
                cased before counting; for example, 'happy' and 'Happy' would be
                grouped together as 'happy', with a count of 2
            normalize (bool): if True, normalize individual words' counts by the
                total token count, giving instead their *relative* frequency of
                occurrence in ``Doc``; note: the resulting set of values won't
                (necessarily) sum to 1.0, since punctuation and stop words are
                filtered out after counts are normalized
            as_strings (bool): if True, words are returned as strings; if False
                (default), words are returned as their unique integer ids
            kwargs:
                - filter_stops (bool)
                - filter_punct (bool)
                - filter_nums (bool)
                - include_pos (str or Set[str])
                - exclude_pos (str or Set[str])
                - min_freq (int)
                - include_types (str or Set[str])
                - exclude_types (str or Set[str]
                - drop_determiners (bool)
                see :func:`extract.words <textacy.extract.words>`,
                :func:`extract.ngrams <textacy.extract.ngrams>`,
                and :func:`extract.named_entities <textacy.extract.named_entities>`
                for more information on these parameters

        Returns:
            dict: mapping of a unique term id or string (depending on the value
                of `as_strings`) to its absolute or relative frequency of
                occurrence (depending on the value of `normalize`)

        .. seealso:: :meth:`Doc.to_terms_list <textacy.document.Doc.to_terms_list>`
        """
        terms_list = self.to_terms_list(
            ngrams=ngrams, named_entities=named_entities,
            lemmatize=lemmatize, lowercase=lowercase,
            as_strings=as_strings, **kwargs)
        bot = itertoolz.frequencies(terms_list)
        if normalize is True:
            n_tokens = self.n_tokens
            bot = {term: weight / n_tokens
                   for term, weight in bot.items()}
        return bot

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

    _counted_ngrams = set()
    _counts = Counter()

    def count(self, term):
        """
        Get the frequency of occurrence ("count") of ``term`` in ``Doc``.

        Args:
            term (str or int or ``spacy.Token`` or ``spacy.Span``): the term to
                be counted can be given as a string, a unique integer id, a
                spacy token, or a spacy span; counts for the same term given in
                different forms will be the same

        Returns:
            int: count of ``term`` in ``Doc``

        .. note:: Counts are cached. The first time a single word's count is
            looked up, *all* words' counts are saved, resulting in a slower
            runtime the first time but orders of magnitude faster runtime for
            subsequent calls for this or any other word. Similarly, if a
            bigram's count is looked up, all bigrams' counts are stored — etc.
        .. warning: If spans are merged using :meth:`Doc.merge() <textacy.Doc.merge>`,
            all cached counts are deleted, since merging spans will invalidate
            many counts. Better to merge first, count second!
        """
        # figure out what object we're dealing with here; convert as necessary
        if isinstance(term, unicode_type):
            term_text = term
            term_id = self.spacy_stringstore[term_text]
            term_len = term_text.count(' ') + 1
        elif isinstance(term, int):
            term_id = term
            term_text = self.spacy_stringstore[term_id]
            term_len = term_text.count(' ') + 1
        elif isinstance(term, SpacyToken):
            term_text = term.orth_
            term_id = self.spacy_stringstore[term_text]
            term_len = 1
        elif isinstance(term, SpacySpan):
            term_text = term.orth_
            term_id = self.spacy_stringstore[term_text]
            term_len = len(term)

        # we haven't counted terms of this length; let's do that now
        if term_len not in self._counted_ngrams:
            if term_len == 1:
                self._counts += Counter(
                    word.orth
                    for word in textacy.extract.words(self,
                                                      filter_stops=False,
                                                      filter_punct=False,
                                                      filter_nums=False))
            else:
                self._counts += Counter(
                    self.spacy_stringstore[ngram.orth_]
                    for ngram in textacy.extract.ngrams(self, term_len,
                                                        filter_stops=False,
                                                        filter_punct=False,
                                                        filter_nums=False))
            self._counted_ngrams.add(term_len)

        count_ = self._counts[term_id]
        return count_
