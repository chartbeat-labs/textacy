"""
Wikipedia Corpus Reader
-----------------------

Stream a corpus of Wikipedia articles saved in standardized database dumps,
as either plaintext strings or structured content + metadata dicts.

When parsed, article pages have the following fields:

    - ``title``: title of the Wikipedia article
    - ``page_id``: unique identifier of the page, usable in Wikimedia APIs
    - ``wiki_links``: a list of other article pages linked to from this page
    - ``ext_links``: a list of external URLs linked to from this page
    - ``categories``: a list of Wikipedia categories to which this page belongs
    - ``sections``: a list of article content and associated metadata split up
      according to the section hierarchy of the page; each section contains:
        - ``text``: text content of the section
        - ``idx``: ordered position on the page, from top (0) to bottom
        - ``level``: level (or depth) in the sections hierarchy

DB dumps are downloadable from https://meta.wikimedia.org/wiki/Data_dumps.
"""
import os
import re

from gensim.corpora.dictionary import Dictionary
from gensim.corpora.wikicorpus import extract_pages, filter_wiki, WikiCorpus

from textacy.compat import PY2
from textacy.fileio import open_sesame

WIKI_NEWLINE_RE = re.compile(r'\n{2,5}')
WIKI_HEADER_RE = re.compile(r'={2,5}(.*?)={2,5}')
WIKI_QUOTE_RE = re.compile(r"'{2,3}")
WIKI_CRUFT_RE = re.compile(r'\n\* ?')


class WikiReader(object):
    """
    Stream Wikipedia pages from standardized, compressed files on disk, either as
    plaintext strings or dict documents with both text content and metadata.
    Download the data from https://meta.wikimedia.org/wiki/Data_dumps.

    .. code-block:: pycon

        >>> wr = WikiReader('/path/to/enwiki-latest-pages-articles.xml.bz2')
        >>> for text in wr.texts(limit=5):  # plaintext pages
        ...     print(text)
        >>> for record in wr.records(min_len=100, limit=1):  # parsed pages
        ...     print(record.keys())
        ...     print(' '.join(section['text'] for section in record['sections']))

    Args:
        path (str): full name of database dump file on disk
    """

    def __init__(self, path):
        self.path = path
        self.wikicorpus = WikiCorpus(path, lemmatize=False, dictionary=Dictionary(),
                                     filter_namespaces={'0'})

    def __repr__(self):
        filepath = os.path.split(self.path)[-1]
        return 'WikiReader("{}")'.format(filepath)

    def __iter__(self):
        """
        Iterate over the pages in a Wikipedia articles database dump (*articles.xml.bz2),
        yielding one (page id, title, page content) 3-tuple at a time.
        """
        if PY2 is False:
            for title, content, page_id in extract_pages(open_sesame(self.wikicorpus.fname, mode='rt'),
                                                         self.wikicorpus.filter_namespaces):
                yield (page_id, title, content)
        else:  # Python 2 sucks and can't open bzip in text mode
            for title, content, page_id in extract_pages(open_sesame(self.wikicorpus.fname, mode='rb'),
                                                         self.wikicorpus.filter_namespaces):
                yield (page_id, title, content)

    def _clean_content(self, content):
        return WIKI_CRUFT_RE.sub(
            r'', WIKI_NEWLINE_RE.sub(
                r'\n', WIKI_HEADER_RE.sub(
                    r'\1', WIKI_QUOTE_RE.sub(
                        r'', filter_wiki(content))))).strip()

    def _parse_content(self, content, parser):
        wikicode = parser.parse(content)
        parsed_page = {'sections': []}

        wikilinks = [str(wc.title) for wc in wikicode.ifilter_wikilinks()]
        parsed_page['categories'] = [wc for wc in wikilinks if wc.startswith('Category:')]
        parsed_page['wiki_links'] = [wc for wc in wikilinks
                                     if not wc.startswith('Category:') and
                                     not wc.startswith('File:') and
                                     not wc.startswith('Image:')]
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
                    sec['text'] = re.sub(r'^' + re.escape(sec['title']) + r'\s*', '', sec['text'])
                parsed_page['sections'].append(sec)
                section_idx += 1

            # dammit! the parser has failed us; let's handle it as best we can
            elif len(headings) > 1:
                titles = [str(h.title).strip() for h in headings]
                levels = [int(h.level) for h in headings]
                sub_sections = [str(ss) for ss in
                                re.split(r'\s*' + '|'.join(re.escape(str(h)) for h in headings) + r'\s*', str(section))]
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
                                                        'text': self._clean_content(sub_section)})
                        section_idx += 1
                    except IndexError:
                        continue

        return parsed_page

    def texts(self, min_len=100, limit=-1):
        """
        Iterate over the pages in a Wikipedia articles database dump
        (``*articles.xml.bz2``), yielding the text of a page, one at a time.

        Args:
            min_len (int): minimum length in chars that a page must have
                for it to be returned; too-short pages are skipped (optional)
            limit (int): maximum number of pages (passing `min_len`) to yield;
                if -1, all pages in the db dump are iterated over (optional)

        Yields:
            str: plain text for the next page in the wikipedia database dump

        Notes:
            Page and section titles appear immediately before the text content
                that they label, separated by a single newline character.
        """
        n_pages = 0
        for _, title, content in self:
            text = self._clean_content(content)
            if len(text) < min_len:
                continue

            yield title + '\n' + text

            n_pages += 1
            if n_pages == limit:
                break

    def records(self, min_len=100, limit=-1):
        """
        Iterate over the pages in a Wikipedia articles database dump
        (``*articles.xml.bz2``), yielding one page whose structure and content
        have been parsed, as a dict.

        Args:
            min_len (int): minimum length in chars that a page must have
                for it to be returned; too-short pages are skipped
            limit (int): maximum number of pages (passing ``min_len``) to yield;
                if -1, all pages in the db dump are iterated over (optional)

        Yields:
            dict: the next page's parsed content, including 'title' and 'page_id'

                Key 'sections' includes a list of all page sections, each with 'title'
                for the section title, 'text' for plain text content,'idx' for position
                on page, and 'level' for the depth of the section within the page's hierarchy

        .. note:: This function requires `mwparserfromhell <mwparserfromhell.readthedocs.org>`_
        """
        try:
            import mwparserfromhell  # hiding this here; don't want another required dep
        except ImportError:
            print('mwparserfromhell package must be installed; see http://pythonhosted.org/mwparserfromhell/')
            raise
        parser = mwparserfromhell.parser.Parser()

        n_pages = 0
        for page_id, title, content in self:
            page = self._parse_content(content, parser)
            if len(' '.join(s['text'] for s in page['sections'])) < min_len:
                continue
            page['title'] = title
            page['page_id'] = page_id

            yield page

            n_pages += 1
            if n_pages == limit:
                break
