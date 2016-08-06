# -*- coding: utf-8 -*-
"""
Load, process, iterate, transform, and save a collection of documents — a corpus.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import os
import warnings

import spacy.about

from textacy import data, fileio
from textacy.compat import PY2, string_types, zip
from textacy.document import Document
from textacy.representations import vsm


class Corpus(object):
    """
    An ordered collection of :class:`Document <textacy.document.Document>` s, all of
    the same language and sharing a single spaCy pipeline and vocabulary. Tracks
    overall corpus statistics and provides a convenient interface to alternate
    corpus representations.

    Args:
        lang_or_pipeline ({'en', 'de'} or :class:`spacy.<lang>.<Language>`):
            either 'en' or 'de', used to initialize the corresponding spaCy pipeline
            with all models loaded by default, or an already-initialized spaCy
            pipeline which may or may not have all models loaded

    Add texts to corpus one-by-one with :meth:`Corpus.add_text() <textacy.corpus.Corpus.add_text>`,
    or all at once with :meth:`Corpus.from_texts() <textacy.corpus.Corpus.from_texts>`.
    Can also add already-instantiated Documents via :meth:`Corpus.add_doc() <textacy.corpus.Corpus.add_doc>`.

    Iterate over corpus docs with ``for doc in Corpus``. Access individual docs
    by index (e.g. ``Corpus[0]`` or ``Corpus[0:10]``) or by boolean condition
    specified by a function (e.g. ``Corpus.get_docs(lambda x: len(x) > 100)``).
    """
    def __init__(self, lang_or_pipeline):
        if isinstance(lang_or_pipeline, string_types):
            self.lang = lang_or_pipeline
            self.spacy_pipeline = data.load_spacy(self.lang)
        else:
            self.spacy_pipeline = lang_or_pipeline
            self.lang = self.spacy_pipeline.lang
        self.spacy_vocab = self.spacy_pipeline.vocab
        self.spacy_stringstore = self.spacy_vocab.strings
        self.docs = []
        self.n_docs = 0
        self.n_sents = 0
        self.n_tokens = 0

    def __repr__(self):
        return 'Corpus({} docs; {} tokens)'.format(self.n_docs, self.n_tokens)

    def __len__(self):
        return self.n_docs

    def __getitem__(self, index):
        return self.docs[index]

    def __delitem__(self, index):
        del self.docs[index]

    def __iter__(self):
        for doc in self.docs:
            yield doc

    def save(self, path, fname_prefix=None, compression=None):
        """
        Save serialized Corpus content and metadata to disk.

        Args:
            path (str): directory on disk where content + metadata will be saved
            fname_prefix (str, optional): prepend standard filenames 'spacy_docs.bin'
                and 'metadatas.json' with additional identifying information
            compression ({'gzip', 'bz2', 'lzma'} or None): type of compression
                used to reduce size of metadatas json file

        .. warn:: If the `spacy.Vocab` object used to save this corpus is not the
            same as the one used to load it, there will be problems! Consequently,
            this functionality is only useful as short-term but not long-term storage.
        """
        if fname_prefix:
            info_fname = os.path.join(path, '_'.join([fname_prefix, 'info.json']))
            meta_fname = os.path.join(path, '_'.join([fname_prefix, 'metadatas.json']))
            docs_fname = os.path.join(path, '_'.join([fname_prefix, 'spacy_docs.bin']))
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
    def load(cls, path, fname_prefix=None, compression=None):
        """
        Load serialized content and metadata from disk, and initialize a Corpus.

        Args:
            path (str): directory on disk where content + metadata are saved
            fname_prefix (str, optional): additional identifying information
                prepended to standard filenames 'spacy_docs.bin' and 'metadatas.json'
                when saving to disk
            compression ({'gzip', 'bz2', 'lzma'} or None): type of compression
                used to reduce size of metadatas json file

        Returns:
            :class:`textacy.Corpus`

        .. warn:: If the `spacy.Vocab` object used to save this corpus is not the
            same as the one used to load it, there will be problems! Consequently,
            this functionality is only useful as short-term but not long-term storage.
        """
        if fname_prefix:
            info_fname = os.path.join(path, '_'.join([fname_prefix, 'info.json']))
            meta_fname = os.path.join(path, '_'.join([fname_prefix, 'metadatas.json']))
            docs_fname = os.path.join(path, '_'.join([fname_prefix, 'spacy_docs.bin']))
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
                Document(spacy_doc, spacy_pipeline=textcorpus.spacy_pipeline,
                         lang=lang, metadata=metadata))
        return textcorpus

    @classmethod
    def from_texts(cls, lang_or_pipeline, texts, metadata=None, n_threads=2, batch_size=1000):
        """
        Convenience function for creating a :class:`Corpus <textacy.corpus.Corpus>`
        from an iterable of text strings.

        Args:
            lang_or_pipeline ({'en', 'de'} or :class:`spacy.<lang>.<Language>`)
            texts (iterable(str))
            metadata (iterable(dict), optional)
            n_threads (int, optional)
            batch_size (int, optional)

        Returns:
            :class:`Corpus <textacy.corpus.Corpus>`
        """
        textcorpus = cls(lang_or_pipeline)
        spacy_docs = textcorpus.spacy_pipeline.pipe(
            texts, n_threads=n_threads, batch_size=batch_size)
        if metadata is not None:
            for spacy_doc, md in zip(spacy_docs, metadata):
                textcorpus.add_doc(Document(spacy_doc, lang=textcorpus.lang,
                                            spacy_pipeline=textcorpus.spacy_pipeline,
                                            metadata=md))
        else:
            for spacy_doc in spacy_docs:
                textcorpus.add_doc(Document(spacy_doc, lang=textcorpus.lang,
                                            spacy_pipeline=textcorpus.spacy_pipeline,
                                            metadata=None))
        return textcorpus

    def add_text(self, text, lang=None, metadata=None):
        """
        Create a :class:`Document <textacy.document.Document>` from ``text`` and ``metadata``,
        then add it to the corpus.

        Args:
            text (str): raw text document to add to corpus as newly instantiated
                :class:`Document <textacy.document.Document>`
            lang (str, optional):
            metadata (dict, optional): dictionary of document metadata, such as::

                {"title": "My Great Doc", "author": "Burton DeWilde"}

                NOTE: may be useful for retrieval via :func:`get_docs() <textacy.corpus.Corpus.get_docs>`,
                e.g. ``Corpus.get_docs(lambda x: x.metadata["title"] == "My Great Doc")``
        """
        doc = Document(text, spacy_pipeline=self.spacy_pipeline,
                       lang=lang, metadata=metadata)
        doc.corpus_index = self.n_docs
        doc.corpus = self
        self.docs.append(doc)
        self.n_docs += 1
        self.n_tokens += doc.n_tokens
        # sentence segmentation requires parse; if not available, just skip this
        try:
            self.n_sents += doc.n_sents
        except ValueError:
            pass

    def add_doc(self, doc, print_warning=True):
        """
        Add an existing :class:`Document <textacy.document.Document>` to the corpus as-is.
        NB: If ``Document`` is already added to this or another :class:`Corpus <textacy.corpus.Corpus>`,
        a warning message will be printed and the ``corpus_index`` attribute will be
        overwritten, but you won't be prevented from adding the doc.

        Args:
            doc (:class:`Document <textacy.document.Document>`)
            print_warning (bool, optional): if True, print a warning message if
                ``doc`` already added to a corpus; otherwise, don't ever print
                the warning and live dangerously
        """
        if doc.lang != self.lang:
            msg = 'Document.lang {} != Corpus.lang {}'.format(doc.lang, self.lang)
            raise ValueError(msg)
        if hasattr(doc, 'corpus_index'):
            doc = copy.deepcopy(doc)
            if print_warning is True:
                print('**WARNING: Document already associated with a Corpus; adding anyway...')
        doc.corpus_index = self.n_docs
        doc.corpus = self
        self.docs.append(doc)
        self.n_docs += 1
        self.n_tokens += doc.n_tokens
        # sentence segmentation requires parse; if not available, just skip this
        try:
            self.n_sents += doc.n_sents
        except ValueError:
            pass

    def get_doc(self, index):
        """
        Get a single doc by its position ``index`` in the corpus.
        """
        return self.docs[index]

    def get_docs(self, match_condition, limit=None):
        """
        Iterate over all docs in corpus and return all (or N = ``limit``) for which
        ``match_condition(doc) is True``.

        Args:
            match_condition (func): function that operates on a :class:`Document`
                and returns a boolean value; e.g. `lambda x: len(x) > 100` matches
                all docs with more than 100 tokens
            limit (int, optional): if not `None`, maximum number of matched docs
                to return

        Yields:
            :class:`Document <textacy.document.Document>`: one per doc passing ``match_condition``
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
        n_tokens_removed = self[index].n_tokens
        try:
            n_sents_removed = self[index].n_sents
        except ValueError:
            n_sents_removed = 0
        del self[index]
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
            match_condition (func): function that operates on a :class:`Document <textacy.document.Document>`
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
            del self[index]
        # now let's re-set the `corpus_index` attribute for all docs at once
        for i, doc in enumerate(self):
            doc.corpus_index = i
        # also decrement the corpus doc/sent/token counts
        self.n_docs -= n_docs_removed
        self.n_sents -= n_sents_removed
        self.n_tokens -= n_tokens_removed

    def as_doc_term_matrix(self, terms_lists, weighting='tf',
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
