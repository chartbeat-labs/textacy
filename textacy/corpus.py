# -*- coding: utf-8 -*-
"""
Load, process, iterate, transform, and save a collection of documents — a corpus.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import os
import warnings

import spacy.about
from spacy.language import Language as SpacyLang
from spacy.tokens.doc import Doc as SpacyDoc

from textacy import data, fileio
from textacy.compat import PY2, unicode_type, zip
from textacy.doc import Doc
from textacy.representations import vsm


class Corpus(object):
    """
    An ordered collection of :class:`Doc <textacy.Doc>` s, all of
    the same language and sharing a single spaCy pipeline and vocabulary. Tracks
    overall corpus statistics and provides a convenient interface to alternate
    corpus representations.

    Args:
        lang (str or ``spacy.Language``):
            either 'en' or 'de', used to initialize the corresponding spaCy pipeline
            with all models loaded by default, or an already-initialized spaCy
            pipeline which may or may not have all models loaded

    Add texts to corpus one-by-one with :meth:`Corpus.add_text() <textacy.corpus.Corpus.add_text>`,
    or all at once with :meth:`Corpus.from_texts() <textacy.corpus.Corpus.from_texts>`.
    Can also add already-instantiated Docs via :meth:`Corpus.add_doc() <textacy.corpus.Corpus.add_doc>`.

    Iterate over corpus docs with ``for doc in Corpus``. Access individual docs
    by index (e.g. ``Corpus[0]`` or ``Corpus[0:10]``) or by boolean condition
    specified by a function (e.g. ``Corpus.get_docs(lambda x: len(x) > 100)``).
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

    def __getitem__(self, index):
        return self.docs[index]

    def __iter__(self):
        for doc in self.docs:
            yield doc

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
        textcorpus = Corpus(lang)
        metadata_stream = fileio.read_json_lines(meta_fname, mode=meta_mode,)
        spacy_docs = fileio.read_spacy_docs(textcorpus.spacy_vocab, docs_fname)
        for spacy_doc, metadata in zip(spacy_docs, metadata_stream):
            textcorpus.add_doc(
                Doc(spacy_doc, spacy_pipeline=textcorpus.spacy_pipeline,
                    lang=lang, metadata=metadata))
        return textcorpus

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
        optionally) in parallel with spaCy; add as :class:`textacy.Doc <Doc>` s
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

        .. seealso: :func:`fileio.split_content_and_metadata()`
        .. seealso: http://spacy.io/docs/#multi-threaded
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
        Create a :class:`textacy.Doc <Doc>` from ``text`` and ``metadata``,
        then add it to the corpus.

        Args:
            text (str): Document (text) content to add to corpus as a ``Doc``.
            metadata (dict): Dictionary of relevant document metadata.
        """
        self._add_textacy_doc(Doc(text, lang=self.spacy_lang, metadata=metadata))

    def add_doc(self, doc, metadata=None):
        """
        Add an existing ``textacy.Doc`` or initialize a new one from a
        ``spacy.Doc`` to the corpus.

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

    def get_doc(self, index):
        """
        Get a single :class:`textacy.Doc <Doc>` by its position ``index`` in
        the corpus.
        """
        return self.docs[index]

    def get_docs(self, match_condition, limit=-1):
        """
        Iterate over all docs in ``Corpus`` and return all (or N <= ``limit``)
        for which ``match_condition(doc)`` is True.

        Args:
            match_condition (func): Function that takes a :class:`textacy.Doc <Doc>`
                as input and returns a boolean value. For example, the function
                `lambda x: len(x) > 100` matches all docs with 100+ tokens.
            limit (int): Maximum number of matched docs to return.

        Yields:
            :class:`Doc <textacy.Doc>`: one per doc passing ``match_condition``
                up to ``limit`` docs

        .. tip:: Document metadata may be useful for retrieving only certain
            docs via :func:`get_docs() <Corpus.get_docs>`. For example::

                Corpus.get_docs(lambda x: x.metadata['author'] == 'Burton DeWilde')
        """
        n_matched_docs = 0
        for doc in self:
            if match_condition(doc) is True:
                yield doc
                n_matched_docs += 1
                if n_matched_docs == limit:
                    break

    ###############
    # REMOVE DOCS #

    def _remove_textacy_doc(self, doc):
        pass  # TODO: this function?

    def remove_doc(self, index):
        """
        Remove the document at ``index`` from the corpus, and decrement the
        ``corpus_index`` attribute on all docs that come after it in the corpus.
        """
        n_tokens_removed = self[index].n_tokens
        try:
            n_sents_removed = self[index].n_sents
        except ValueError:
            n_sents_removed = 0
        del self.docs[index]
        # reset `corpus_index` attribute on docs higher up in the list
        for doc in self[index:]:
            doc.corpus_index -= 1
        # also decrement the corpus doc/sent/token counts
        self.n_docs -= 1
        self.n_sents -= n_sents_removed
        self.n_tokens -= n_tokens_removed

    def remove_docs(self, match_condition, limit=None):
        """
        Remove all (or N = ``limit``) docs in corpus for which ``match_condition(doc) is True``.
        Re-set all remaining docs' ``corpus_index`` attributes at the end.

        Args:
            match_condition (func): function that operates on a :class:`Doc <textacy.Doc>`
                and returns a boolean value; e.g. ``lambda x: len(x) > 100`` matches
                all docs with more than 100 tokens
            limit (int, optional): if not None, maximum number of matched docs
                to remove
        """
        remove_indexes = sorted(
            (doc.corpus_index
             for doc in self.get_docs(match_condition, limit=limit)),
            reverse=True)
        n_docs_removed = len(remove_indexes)
        n_sents_removed = 0
        n_tokens_removed = 0
        for index in remove_indexes:
            n_tokens_removed += self[index].n_tokens
            try:
                n_sents_removed += self[index].n_sents
            except ValueError:
                pass
            del self.docs[index]
        # now let's re-set the `corpus_index` attribute for all docs at once
        for i, doc in enumerate(self):
            doc.corpus_index = i
        # also decrement the corpus doc/sent/token counts
        self.n_docs -= n_docs_removed
        self.n_sents -= n_sents_removed
        self.n_tokens -= n_tokens_removed

    ####################
    # TRANSFORM CORPUS #

    def to_doc_term_matrix(self, terms_lists, weighting='tf',
                           normalize=True, smooth_idf=True, sublinear_tf=False,
                           min_df=1, max_df=1.0, min_ic=0.0, max_n_terms=None):
        """
        Transform corpus into a sparse CSR matrix, where each row i corresponds
        to a doc, each column j corresponds to a unique term, and matrix values
        (i, j) correspond to the tf or tf-idf weighting of term j in doc i.

        .. seealso:: :func:`build_doc_term_matrix <textacy.representations.vsm.build_doc_term_matrix>`
        """
        self.doc_term_matrix, self.id_to_term = vsm.build_doc_term_matrix(
            terms_lists, weighting=weighting,
            normalize=normalize, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf,
            min_df=min_df, max_df=max_df, min_ic=min_ic,
            max_n_terms=max_n_terms)
        return (self.doc_term_matrix, self.id_to_term)
