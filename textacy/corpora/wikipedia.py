"""
Wikipedia Corpus
----------------

Functions to read, iterate over, and extract text and/or structured information
from Wikipedia article database dumps or the Wikimedia API.
"""
import bz2
import re

from gensim.corpora.dictionary import Dictionary
from gensim.corpora.wikicorpus import extract_pages, filter_wiki, WikiCorpus

WIKI_NEWLINE_RE = re.compile(r'\n{2,5}')
WIKI_HEADER_RE = re.compile(r'={2,5}(.*?)={2,5}')
WIKI_QUOTE_RE = re.compile(r"'{2,3}")
WIKI_CRUFT_RE = re.compile(r'\n\* ?')


def _iterate_over_pages(fname):
    """
    Iterate over the pages in a Wikipedia articles database dump (*articles.xml.bz2),
    yielding one (page id, title, page content) 3-tuple at a time.
    """
    dictionary = Dictionary()
    wiki = WikiCorpus(fname, lemmatize=False, dictionary=dictionary,
                      filter_namespaces={'0'})
    for title, content, page_id in extract_pages(bz2.BZ2File(wiki.fname), wiki.filter_namespaces):
        yield (page_id, title, content)


def _get_plaintext(content):
    return WIKI_CRUFT_RE.sub(
        r'', WIKI_NEWLINE_RE.sub(
            r'\n', WIKI_HEADER_RE.sub(
                r'\1', WIKI_QUOTE_RE.sub(
                    r'', filter_wiki(content))))).strip()


def _parse_content(content, parser, metadata=True):
    wikicode = parser.parse(content)
    parsed_page = {'sections': []}

    if metadata is True:
        wikilinks = [str(wc.title) for wc in wikicode.ifilter_wikilinks()]
        parsed_page['categories'] = [wc for wc in wikilinks if wc.startswith('Category:')]
        parsed_page['wiki_links'] = [wc for wc in wikilinks
                                     if not wc.startswith('Category:')
                                     and not wc.startswith('File:')
                                     and not wc.startswith('Image:')]
        parsed_page['ext_links'] = [str(wc.url) for wc in wikicode.ifilter_external_links()]

    def _filter_tags(obj):
        return obj.tag == 'ref' or obj.tag == 'table'

    bad_section_titles = {'external links', 'notes', 'references'}
    section_idx = 0

    for section in wikicode.get_sections(flat=True, include_lead=True, include_headings=True):
        headings = section.filter_headings()
        sec = {'idx': section_idx}

        if section_idx == 0 or len(headings) == 1:
            try:
                sec_title = str(headings[0].title)
                if sec_title.lower() in bad_section_titles:
                    continue
                sec['title'] = sec_title
                sec['level'] = int(headings[0].level)
            except IndexError:
                if section_idx == 0:
                    sec['level'] = 1
            # strip out references, tables, and file/image links
            for obj in section.ifilter_tags(matches=_filter_tags, recursive=True):
                section.remove(obj)
            for obj in section.ifilter_wikilinks(recursive=True):
                try:
                    obj_title = str(obj.title)
                    if obj_title.startswith('File:') or obj_title.startswith('Image:'):
                        section.remove(obj)
                except Exception:
                    pass
            sec['text'] = str(section.strip_code(normalize=True, collapse=True)).strip()
            if sec.get('title'):
                sec['text'] = re.sub(r'^'+re.escape(sec['title'])+r'\s*', '', sec['text'])
            parsed_page['sections'].append(sec)
            section_idx += 1

        # dammit! the parser has failed us; let's handle it as best we can
        elif len(headings) > 1:
            titles = [str(h.title).strip() for h in headings]
            levels = [int(h.level) for h in headings]
            sub_sections = [str(ss) for ss in
                            re.split(r'\s*'+'|'.join(re.escape(str(h)) for h in headings)+r'\s*', str(section))]
            # re.split leaves an empty string result up front :shrug:
            if sub_sections[0] == '':
                del sub_sections[0]
            if len(headings) != len(sub_sections):
                print('# headings =', len(headings), '# sections =', len(sub_sections))
            for i, sub_section in enumerate(sub_sections):
                try:
                    if titles[i].lower() in bad_section_titles:
                        continue
                    parsed_page['sections'].append({'title': titles[i], 'level': levels[i], 'idx': section_idx,
                                                    'text': _get_plaintext(sub_section)})
                    section_idx += 1
                except IndexError:
                    continue

    return parsed_page


def get_plaintext_pages(fname, min_len=100, max_n_pages=None):
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
    n_pages = 0
    for page_id, title, content in _iterate_over_pages(fname):
        plaintext = _get_plaintext(content)
        if len(plaintext) < min_len:
            continue
        n_pages += 1
        if max_n_pages is not None and n_pages > max_n_pages:
            break

        yield (page_id, title, plaintext)


def get_parsed_pages(fname, min_len=100, max_n_pages=None, metadata=True):
    """
    Iterate over the pages in a Wikipedia articles database dump (*articles.xml.bz2),
    yielding one page whose structure and content have been parsed, as a dict.

    Args:
        fname (str): full path/to/file for the database dump
        min_len (int, optional): minimum length in chars that a page must have
            for it to be returned; too-short pages are skipped
        max_n_pages (int, optional): maximum number of pages (passing ``min_len``)
            to yield; if None, all pages in the db dump are iterated over
        metadata (bool, optional): if True, return page metadata in addition to content,
            including 'ext_links', 'wiki_links', and 'categories'

    Yields:
        dict: the next page's parsed content, including 'title' and 'page_id'

            Key 'sections' includes a list of all page sections, each with 'title'
            for the section title, 'text' for plain text content,'idx' for position
            on page, and 'level' for the depth of the section within the page's hierarchy

            If ``metadata`` is True, there are additional keys: 'wiki_links' is
            a list of _other_ page titles linked to from this page; 'ext_links' is
            a list of external URLs linked to from this page; and 'categories' is
            a list of Wikipedia categories to which this page belongs.

    Notes:
        .. This function requires `mwparserfromhell <mwparserfromhell.readthedocs.org>`_
    """
    import mwparserfromhell  # hiding this here; don't want another required dep
    parser = mwparserfromhell.parser.Parser()

    n_pages = 0
    for page_id, title, content in _iterate_over_pages(fname):
        parsed_page = _parse_content(content, parser, metadata=metadata)
        if len(' '.join(s['text'] for s in parsed_page['sections'])) < min_len:
            continue
        n_pages += 1
        parsed_page['title'] = title
        parsed_page['page_id'] = page_id
        if max_n_pages is not None and n_pages > max_n_pages:
            break

        yield parsed_page
