# -*- coding: utf-8 -*-
"""
Load, process, iterate, transform, and save a collection of documents — a corpus.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from collections import Counter
import copy
from math import log
import os
import warnings

import spacy.about
from spacy.language import Language as SpacyLang
from spacy.tokens.doc import Doc as SpacyDoc

from textacy import data, fileio
from textacy.compat import PY2, unicode_type, zip
from textacy.doc import Doc


class Corpus(object):
    """
    An ordered collection of :class:`textacy.Doc <textacy.Doc>` s, all of the
    same language and sharing the same ``spacy.Language`` models and vocabulary.
    Track corpus statistics; flexibly add, iterate through, filter for, and
    remove documents; save and load parsed content and metadata to and from
    disk; and more.

    Initialize from a stream of texts and corresponding metadatas::

        >>> cw = textacy.corpora.CapitolWords()
        >>> records = cw.docs(limit=50)
        >>> text_stream, metadata_stream = textacy.fileio.split_record_fields(
        ...     records, 'text')
        >>> corpus = textacy.Corpus(
        ...     'en', texts=text_stream, metadatas=metadata_stream)
        >>> print(corpus)
        Corpus(50 docs; 32163 tokens)

    Index, slice, and flexibly get particular documents::

        >>> corpus[0]
        Doc(159 tokens; "Mr. Speaker, 480,000 Federal employees are work...")
        >>> corpus[:3]
        [Doc(159 tokens; "Mr. Speaker, 480,000 Federal employees are work..."),
         Doc(219 tokens; "Mr. Speaker, a relationship, to work and surviv..."),
         Doc(336 tokens; "Mr. Speaker, I thank the gentleman for yielding...")]
        >>> match_func = lambda doc: doc.metadata['speaker_name'] == 'Bernie Sanders'
        >>> for doc in corpus.get(match_func, limit=3):
        ...     print(doc)
        Doc(159 tokens; "Mr. Speaker, 480,000 Federal employees are work...")
        Doc(336 tokens; "Mr. Speaker, I thank the gentleman for yielding...")
        Doc(177 tokens; "Mr. Speaker, if we want to understand why in th...")

    Add and remove documents, with automatic updating of corpus statistics::

        >>> records = cw.docs(congress=114, limit=25)
        >>> text_stream, metadata_stream = textacy.fileio.split_record_fields(
        ...     records, 'text')
        >>> corpus.add_texts(text_stream, metadatas=metadata_stream, n_threads=4)
        >>> print(corpus)
        Corpus(75 docs; 55869 tokens)
        >>> corpus.remove(lambda doc: doc.metadata['speaker_name'] == 'Rick Santorum')
        >>> print(corpus)
        Corpus(60 docs; 48532 tokens)
        >>> del corpus[:5]
        >>> print(corpus)
        Corpus(55 docs; 47444 tokens)

    Get word and doc frequencies in absolute, relative, or binary form:

        >>> counts = corpus.word_freqs(lemmatize=True, weighting='count')
        >>> idf = corpus.word_doc_freqs(lemmatize=True, weighting='idf')

    Save to and load from disk::

        >>> corpus.save('~/Desktop', name='congress', compression='gzip')
        >>> corpus = textacy.Corpus.load(
        ...     '~/Desktop', name='congress', compression='gzip')
        >>> print(corpus)
        Corpus(55 docs; 47444 tokens)

    Args:
        lang (str or ``spacy.Language``): Language of content for all docs in
            corpus as a 2-letter code, used to instantiate the corresponding
            ``spacy.Language`` with all models loaded by default, or an already-
            instantiated ``spacy.Language`` which may or may not have all models
            loaded, depending on the user's needs. Currently, spaCy only handles
            English ('en') and German ('de') text.
        texts (Iterable[str]): Stream of documents as (unicode) text, to be
            processed by spaCy and added to the corpus as :class:`textacy.Doc <textacy.Doc>`s.
        docs (Iterable[``textacy.Doc``] or Iterable[``spacy.Doc``]): Stream of
            documents already-processed by spaCy alone or via textacy.
        metadatas (Iterable[dict]): Stream of dictionaries of relevant doc
            metadata. **Note:** This stream must align exactly with ``texts`` or
            ``docs``, or else metadata will be mis-assigned. More concretely,
            the first item in ``metadatas`` will be assigned to the first item
            in ``texts`` or ``docs``, and so on from there.

    Attributes:
        lang (str): 2-letter code for language of documents in ``Corpus``.
        n_docs (int): Number of documents in ``Corpus``.
        n_tokens (int): Total number of tokens of all documents in ``Corpus``.
        n_sents (int): Total number of sentences of all documents in ``Corpus``.
            If the ``spacy.Language`` used to process documents did not include
            a syntactic parser, upon which sentence segmentation relies, this
            value will be null.
        docs (List[``textacy.Doc``]): List of documents in ``Corpus``. In 99\%
            of cases, you should never have to interact directly with this list;
            instead, index and slice directly on ``Corpus`` or use the flexible
            :meth:`Corpus.get() <Corpus.get>` and :meth:`Corpus.remove() <Corpus.remove`
            methods.
        spacy_lang (``spacy.Language``): http://spacy.io/docs/#english
        spacy_vocab (``spacy.Vocab``): https://spacy.io/docs#vocab
        spacy_stringstore (``spacy.StringStore``): https://spacy.io/docs#stringstore
    """
    def __init__(self, lang, texts=None, docs=None, metadatas=None):
        if isinstance(lang, unicode_type):
            self.lang = lang
            self.spacy_lang = data.load_spacy(self.lang)
        elif isinstance(lang, SpacyLang):
            self.lang = lang.lang
            self.spacy_lang = lang
        else:
            msg = '`lang` must be {}, not "{}"'.format(
                {unicode_type, SpacyLang}, type(lang))
            raise ValueError(msg)
        self.spacy_vocab = self.spacy_lang.vocab
        self.spacy_stringstore = self.spacy_vocab.strings
        self.docs = []
        self.n_docs = 0
        self.n_tokens = 0
        self.n_sents = 0 if self.spacy_lang.parser else None

        if texts and docs:
            msg = 'Corpus may be initialized with either `texts` or `docs`, but not both.'
            raise ValueError(msg)
        if texts:
            self.add_texts(texts, metadatas=metadatas)
        elif docs:
            if metadatas:
                for doc, metadata in zip(docs, metadatas):
                    self.add_doc(doc, metadata=metadata)
            else:
                for doc in docs:
                    self.add_doc(doc)

    def __repr__(self):
        return 'Corpus({} docs; {} tokens)'.format(self.n_docs, self.n_tokens)

    def __len__(self):
        return self.n_docs

    def __iter__(self):
        for doc in self.docs:
            yield doc

    def __getitem__(self, idx_or_slice):
        return self.docs[idx_or_slice]

    def __delitem__(self, idx_or_slice):
        if isinstance(idx_or_slice, int):
            self._remove_one_doc_by_index(idx_or_slice)
        elif isinstance(idx_or_slice, slice):
            start, end, step = idx_or_slice.indices(self.n_docs)
            indexes = range(start, end, step)
            self._remove_many_docs_by_index(indexes)
        else:
            msg = 'value must be {}, not "{}"'.format(
                {int, slice}, type(idx_or_slice))
            raise ValueError(msg)

    ##########
    # FILEIO #

    def save(self, path, name=None, compression=None):
        """
        Save ``Corpus`` content and metadata to disk.

        Args:
            path (str): Directory on disk where content + metadata will be saved.
            name (str): Prepend default filenames 'spacy_docs.bin', 'metadatas.json',
                and 'info.json' with a name to identify/uniquify this particular
                corpus.
            compression ({'gzip', 'bz2', 'lzma'} or None): Type of compression
                used to reduce size of 'metadatas.json' file, if any.

        .. warning:: If the ``spacy.Vocab`` object used to save this corpus is
            not the same as the one used to load it, there will be problems!
            Consequently, this functionality is only useful as short-term but
            not long-term storage.
        """
        if name:
            info_fname = os.path.join(path, '_'.join([name, 'info.json']))
            meta_fname = os.path.join(path, '_'.join([name, 'metadatas.json']))
            docs_fname = os.path.join(path, '_'.join([name, 'spacy_docs.bin']))
        else:
            info_fname = os.path.join(path, 'info.json')
            meta_fname = os.path.join(path, 'metadatas.json')
            docs_fname = os.path.join(path, 'spacy_docs.bin')
        meta_fname = meta_fname + ('.gz' if compression == 'gzip'
                                   else '.bz2' if compression == 'bz2'
                                   else '.xz' if compression == 'lzma'
                                   else '')
        meta_mode = 'wt' if PY2 is False or compression is None else 'wb'
        package_info = {'textacy_lang': self.lang, 'spacy_version': spacy.about.__version__}
        fileio.write_json(package_info, info_fname)
        fileio.write_json_lines(
            (doc.metadata for doc in self), meta_fname, mode=meta_mode,
            ensure_ascii=False, separators=(',', ':'))
        fileio.write_spacy_docs((doc.spacy_doc for doc in self), docs_fname)

    @classmethod
    def load(cls, path, name=None, compression=None):
        """
        Load content and metadata from disk, and initialize a ``Corpus``.

        Args:
            path (str): Directory on disk where content + metadata are saved.
            name (str): Identifying/uniquifying name prepended to the default
                filenames 'spacy_docs.bin', 'metadatas.json', and 'info.json',
                used when corpus was saved to disk via :meth:`Corpus.save()`.
            compression ({'gzip', 'bz2', 'lzma'} or None): Type of compression
                used to reduce size of 'metadatas.json' file when saved, if any.

        Returns:
            :class:`textacy.Corpus <Corpus>`

        .. warning:: If the ``spacy.Vocab`` object used to save this document is
            not the same as the one used to load it, there will be problems!
            Consequently, this functionality is only useful as short-term but
            not long-term storage.
        """
        if name:
            info_fname = os.path.join(path, '_'.join([name, 'info.json']))
            meta_fname = os.path.join(path, '_'.join([name, 'metadatas.json']))
            docs_fname = os.path.join(path, '_'.join([name, 'spacy_docs.bin']))
        else:
            info_fname = os.path.join(path, 'info.json')
            meta_fname = os.path.join(path, 'metadatas.json')
            docs_fname = os.path.join(path, 'spacy_docs.bin')
        meta_fname = meta_fname + ('.gz' if compression == 'gzip'
                                   else '.bz2' if compression == 'bz2'
                                   else '.xz' if compression == 'lzma'
                                   else '')
        meta_mode = 'rt' if PY2 is False or compression is None else 'rb'
        package_info = list(fileio.read_json(info_fname))[0]
        lang = package_info['textacy_lang']
        spacy_version = package_info['spacy_version']
        if spacy_version != spacy.about.__version__:
            msg = """
                the spaCy version used to save this Corpus to disk is not the
                same as the version currently installed ('{}' vs. '{}'); if the
                data underlying the associated `spacy.Vocab` has changed, this
                loaded Corpus may not be valid!
                """.format(spacy_version, spacy.about.__version__)
            warnings.warn(msg, UserWarning)
        corpus = Corpus(lang)
        metadata_stream = fileio.read_json_lines(meta_fname, mode=meta_mode)
        spacy_docs = fileio.read_spacy_docs(corpus.spacy_vocab, docs_fname)
        for spacy_doc, metadata in zip(spacy_docs, metadata_stream):
            corpus.add_doc(
                Doc(spacy_doc, lang=corpus.spacy_lang, metadata=metadata))
        return corpus

    #################
    # ADD DOCUMENTS #

    def _add_textacy_doc(self, doc):
        doc.corpus_index = self.n_docs
        doc.corpus = self
        self.docs.append(doc)
        self.n_docs += 1
        self.n_tokens += doc.n_tokens
        # sentence segmentation requires parse; if not available, skip it
        if self.spacy_lang.parser:
            self.n_sents += doc.n_sents

    def add_texts(self, texts, metadatas=None, n_threads=4, batch_size=1000):
        """
        Process a stream of texts (and a corresponding stream of metadata dicts,
        optionally) in parallel with spaCy; add as :class:`textacy.Doc <textacy.doc.Doc>` s
        to the corpus.

        Args:
            texts (Iterable[str]): Stream of texts to add to corpus as ``Doc`` s
            metadatas (Iterable[dict]): Stream of dictionaries of relevant
                document metadata. **Note:** This stream must align exactly with
                ``texts``, or metadata will be mis-assigned to texts. More
                concretely, the first item in ``metadatas`` will be assigned to
                the first item in ``texts``, and so on from there.
            n_threads (int): Number of threads to use when processing ``texts``
                in parallel, if available.
            batch_size (int): Number of texts to process at a time.

        See Also:
            :func:`fileio.split_record_fields()`
            http://spacy.io/docs/#multi-threaded
        """
        spacy_docs = self.spacy_lang.pipe(
            texts, n_threads=n_threads, batch_size=batch_size)
        if metadatas:
            for spacy_doc, metadata in zip(spacy_docs, metadatas):
                self._add_textacy_doc(
                    Doc(spacy_doc, lang=self.spacy_lang, metadata=metadata))
        else:
            for spacy_doc in spacy_docs:
                self._add_textacy_doc(
                    Doc(spacy_doc, lang=self.spacy_lang, metadata=None))

    def add_text(self, text, metadata=None):
        """
        Create a :class:`textacy.Doc <textacy.doc.Doc>` from ``text`` and
        ``metadata``, then add it to the corpus.

        Args:
            text (str): Document (text) content to add to corpus as a ``Doc``.
            metadata (dict): Dictionary of relevant document metadata.
        """
        self._add_textacy_doc(Doc(text, lang=self.spacy_lang, metadata=metadata))

    def add_doc(self, doc, metadata=None):
        """
        Add an existing :class:`textacy.Doc <textacy.doc.Doc>` or initialize a
        new one from a ``spacy.Doc`` to the corpus.

        Args:
            doc (``textacy.Doc`` or ``spacy.Doc``)
            metadata (dict): Dictionary of relevant document metadata. If ``doc``
                is a ``spacy.Doc``, it will be paired as usual; if ``doc`` is a
                ``textacy.Doc``, it will *overwrite* any existing metadata.

        .. warning:: If ``doc`` was already added to this or another ``Corpus``,
            it will be deep-copied and then added as if a new document. A warning
            message will be logged. This is probably not a thing you should do.
        """
        if isinstance(doc, Doc):
            if doc.spacy_vocab is not self.spacy_vocab:
                msg = 'Doc.spacy_vocab {} != Corpus.spacy_vocab {}'.format(
                    doc.spacy_vocab, self.spacy_vocab)
                raise ValueError(msg)
            if hasattr(doc, 'corpus_index'):
                doc = copy.deepcopy(doc)
                # TODO: make this into a logging warning
                print('**WARNING: Doc already associated with a Corpus; adding anyway...')
            if metadata is not None:
                doc.metadata = metadata
            self._add_textacy_doc(doc)
        elif isinstance(doc, SpacyDoc):
            if doc.vocab is not self.spacy_vocab:
                msg = 'SpacyDoc.vocab {} != Corpus.spacy_vocab {}'.format(
                    doc.vocab, self.spacy_vocab)
                raise ValueError(msg)
            self._add_textacy_doc(
                Doc(doc, lang=self.spacy_lang, metadata=metadata))
        else:
            msg = '`doc` must be {}, not "{}"'.format(
                {Doc, SpacyDoc}, type(doc))
            raise ValueError(msg)

    #################
    # GET DOCUMENTS #

    def get(self, match_func, limit=-1):
        """
        Iterate over docs in ``Corpus`` and return all (or N <= ``limit``) for
        which ``match_func(doc)`` is True.

        Args:
            match_func (func): Function that takes a :class:`textacy.Doc <Doc>`
                as input and returns a boolean value. For example::

                    Corpus.get(lambda x: len(x) >= 100)

                gets all docs with 100+ tokens. And::

                    Corpus.get(lambda x: x.metadata['author'] == 'Burton DeWilde')

                gets all docs whose author was given as 'Burton DeWilde'.
            limit (int): Maximum number of matched docs to return.

        Yields:
            :class:`textacy.Doc <textacy.doc.Doc>`: next document passing
                ``match_func`` up to ``limit`` docs

        See Also:
            :meth:`Corpus.remove()`

        .. tip:: To get doc(s) by index, treat ``Corpus`` as a list and use
            Python's usual indexing and slicing: ``Corpus[0]`` gets the first
            document in the corpus; ``Corpus[:5]`` gets the first 5; etc.
        """
        n_matched_docs = 0
        for doc in self:
            if match_func(doc) is True:
                yield doc
                n_matched_docs += 1
                if n_matched_docs == limit:
                    break

    ###############
    # REMOVE DOCS #

    def _remove_one_doc_by_index(self, index):
        doc = self[index]
        n_tokens_removed = doc.n_tokens
        n_sents_removed = doc.n_sents if self.spacy_lang.parser else None
        # actually remove the doc
        del self.docs[index]
        # shift `corpus_index` attribute on docs higher up in the list
        for doc in self[index:]:
            doc.corpus_index -= 1
        # decrement the corpus doc/token/sent counts
        self.n_docs -= 1
        self.n_tokens -= n_tokens_removed
        if n_sents_removed:
            self.n_sents -= n_sents_removed

    def _remove_many_docs_by_index(self, indexes):
        indexes = sorted(indexes, reverse=True)
        n_docs_removed = len(indexes)
        n_sents_removed = 0
        n_tokens_removed = 0
        for index in indexes:
            n_tokens_removed += self[index].n_tokens
            if self.spacy_lang.parser:
                n_sents_removed += self[index].n_sents
            # actually remove the doc
            del self.docs[index]
        # shift the `corpus_index` attribute for all docs at once
        for i, doc in enumerate(self):
            doc.corpus_index = i
        # decrement the corpus doc/sent/token counts
        self.n_docs -= n_docs_removed
        self.n_tokens -= n_tokens_removed
        self.n_sents -= n_sents_removed

    def remove(self, match_func, limit=-1):
        """
        Remove all (or N <= ``limit``) docs in ``Corpus`` for which
        ``match_func(doc)`` is True. Corpus doc/sent/token counts are adjusted
        accordingly, as are the :attr:`Doc.corpus_index <textacy.doc.Doc.corpus_index>`
        attributes on affected documents.

        Args:
            match_func (func): Function that takes a :class:`textacy.Doc <textacy.doc.Doc>`
                and returns a boolean value. For example::

                    Corpus.remove(lambda x: len(x) >= 100)

                removes docs with 100+ tokens. And::

                    Corpus.remove(lambda x: x.metadata['author'] == 'Burton DeWilde')

                removes docs whose author was given as 'Burton DeWilde'.
            limit (int): Maximum number of matched docs to remove.

        See Also:
            :meth:`Corpus.get() <Corpus.get>`

        .. tip:: To remove doc(s) by index, treat ``Corpus`` as a list and use
            Python's usual indexing and slicing: ``del Corpus[0]`` removes the
            first document in the corpus; ``del Corpus[:5]`` removes the first
            5; etc.
        """
        n_matched_docs = 0
        matched_indexes = []
        for i, doc in enumerate(self):
            if match_func(doc) is True:
                matched_indexes.append(i)
                n_matched_docs += 1
                if n_matched_docs == limit:
                    break
        self._remove_many_docs_by_index(matched_indexes)

    def word_freqs(self, lemmatize=True, lowercase=False,
                   weighting='count', as_strings=False):
        """
        Map the set of unique words in ``Corpus`` to their counts as absolute,
        relative, or binary frequencies of occurence. This is akin to
        :func:``Doc.to_bag_of_words() <textacy.doc.Doc.to_bag_of_words>.

        Args:
            lemmatize (bool): if True, words are lemmatized before counting;
                for example, 'happy', 'happier', and 'happiest' would be grouped
                together as 'happy', with a count of 3
            lowercase (bool): if True and ``lemmatize`` is False, words are lower-
                cased before counting; for example, 'happy' and 'Happy' would be
                grouped together as 'happy', with a count of 2
            weighting ({'count', 'freq', 'binary'}): Type of weight to assign to
                words. If 'count' (default), weights are the absolute number of
                occurrences (count) of word in corpus. If 'binary', all counts
                are set equal to 1. If 'freq', word counts are normalized by the
                total token count, giving their relative frequency of occurrence.
                Note: The resulting set of frequencies won't (necessarily) sum
                to 1.0, since punctuation and stop words are filtered out after
                counts are normalized.
            as_strings (bool): if True, words are returned as strings; if False
                (default), words are returned as their unique integer ids

        Returns:
            dict: mapping of a unique word id or string (depending on the value
                of ``as_strings``) to its absolute, relative, or binary frequency
                of occurrence (depending on the value of ``weighting``).

        See Also:
            :func:`vsm.get_term_freqs() <textacy.vsm.get_term_freqs>``
        """
        word_counts = Counter()
        for doc in self:
            word_counts.update(doc.to_bag_of_words(
                lemmatize=lemmatize, lowercase=lowercase,
                weighting='count', as_strings=as_strings))
        if weighting == 'count':
            word_counts = dict(word_counts)
        if weighting == 'freq':
            n_tokens = self.n_tokens
            word_counts = {word: weight / n_tokens
                           for word, weight in word_counts.items()}
        elif weighting == 'binary':
            word_counts = {word: 1 for word in word_counts.keys()}
        return word_counts

    def word_doc_freqs(self, lemmatize=True, lowercase=False,
                       weighting='count', smooth_idf=True, as_strings=False):
        """
        Map the set of unique words in ``Corpus`` to their *document* counts as
        absolute, relative, inverse, or binary frequencies of occurence.

        Args:
            lemmatize (bool): if True, words are lemmatized before counting;
                for example, 'happy', 'happier', and 'happiest' would be grouped
                together as 'happy', with a count of 3
            lowercase (bool): if True and ``lemmatize`` is False, words are lower-
                cased before counting; for example, 'happy' and 'Happy' would be
                grouped together as 'happy', with a count of 2
            weighting ({'count', 'freq', 'idf', 'binary'}): Type of weight to
                assign to words. If 'count' (default), weights are the absolute
                number (count) of documents in which word appears. If 'binary',
                all counts are set equal to 1. If 'freq', word doc counts are
                normalized by the total document count, giving their relative
                frequency of occurrence. If 'idf', weights are the log of the
                inverse relative frequencies: ``log(n_docs / word_doc_count)``
                or ``log(1 + n_docs / word_doc_count)`` if ``smooth_idf`` is True.
            smooth_idf (bool): if True, add 1 to all document frequencies when
                calculating 'idf' weighting, equivalent to adding a single
                document to the corpus containing every unique word
            as_strings (bool): if True, words are returned as strings; if False
                (default), words are returned as their unique integer ids

        Returns:
            dict: mapping of a unique word id or string (depending on the value
                of ``as_strings``) to the number of documents in which it appears
                weighted as absolute, relative, or binary frequencies (depending
                on the value of ``weighting``).

        See Also:
            :func:`vsm.get_doc_freqs() <textacy.vsm.get_doc_freqs>``
        """
        word_doc_counts = Counter()
        for doc in self:
            word_doc_counts.update(doc.to_bag_of_words(
                lemmatize=lemmatize, lowercase=lowercase,
                weighting='binary', as_strings=as_strings))
        if weighting == 'count':
            word_doc_counts = dict(word_doc_counts)
        elif weighting == 'freq':
            n_docs = self.n_docs
            word_doc_counts = {word: count / n_docs
                               for word, count in word_doc_counts.items()}
        elif weighting == 'idf':
            n_docs = self.n_docs
            if smooth_idf is True:
                word_doc_counts = {word: log(1 + n_docs / count)
                                   for word, count in word_doc_counts.items()}
            else:
                word_doc_counts = {word: log(n_docs / count)
                                   for word, count in word_doc_counts.items()}
        elif weighting == 'binary':
            word_doc_counts = {word: 1 for word in word_doc_counts.keys()}
        return word_doc_counts
