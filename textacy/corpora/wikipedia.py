"""
Functions to read, iterate over, and extract text and/or structured information
from Wikipedia article database dumps or the Wikimedia API.
"""
import bz2
import re

from gensim.corpora.dictionary import Dictionary
from gensim.corpora.wikicorpus import extract_pages, filter_wiki, WikiCorpus


def _iterate_over_pages(fname):
    """
    Iterate over the pages in a Wikipedia articles database dump (*articles.xml.bz2),
    yielding one (page id, title, page content) 3-tuple at a time.
    """
    dictionary = Dictionary()
    wiki = WikiCorpus(fname, lemmatize=False, dictionary=dictionary)
    for title, content, page_id in extract_pages(bz2.BZ2File(wiki.fname), wiki.filter_namespaces):
        yield (page_id, title, content)


def get_pages_plaintext(fname, min_len=100, max_n_pages=None):
    """
    Iterate over the pages in a Wikipedia articles database dump (*articles.xml.bz2),
    yielding one (page id, title, plaintext) 3-tuple at a time.

    Args:
        fname (str): full path/to/file for the database dump
        min_len (int, optional): minimum length in chars that a page must have
            for it to be returned; too-short pages are skipped
        max_n_pages (int, optional): maximum number of pages (passing ``min_len``)
            to yield; if None, all pages in the db dump are iterated over

    Yields:
        tuple(str, str, str): the next (page id, title, plaintext) 3-tuple
            from the wikipedia articles database dump
    """
    newline_re = re.compile(r'\n{2,5}')
    header_re = re.compile(r'={2,5}(.*?)={2,5}')
    quotation_re = re.compile(r"'{2,3}")
    cruft_re = re.compile(r'\n\* ?')

    n_pages = 0
    for page_id, title, content in _iterate_over_pages(fname):
        plaintext = cruft_re.sub(
            r'', newline_re.sub(
                r'\n', header_re.sub(
                    r'\1', quotation_re.sub(
                        r'', filter_wiki(content)))))
        if len(plaintext) < min_len:
            continue
        n_pages += 1
        if max_n_pages is not None and n_pages > max_n_pages:
            break

        yield (page_id, title, plaintext)
