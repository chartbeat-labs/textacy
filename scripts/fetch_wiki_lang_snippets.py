#!/usr/bin/env python
"""
Heavily inspired by the WiLI-2018 dataset and corresponding code base,
https://github.com/MartinThoma/lidtk.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import datetime
import io
import logging
import os
import re
import sys
import time
import warnings

import wikipedia
from cld2 import detect as cld2_detect
from tqdm import tqdm

# configuring separate stream and file handlers doesn't work
# *seems* like a python bug, but more likely i'm doing something wrong
# so, fuck it, we'll just use the basic config and ignore debugging messages
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

# wikipedia was raising requests warnings; let's ignore
warnings.filterwarnings("ignore")

re_sections = re.compile(r"^={2,6}.*?={2,6}$", flags=re.UNICODE | re.MULTILINE)
re_breaking_whitespace = re.compile(r"\n+", flags=re.UNICODE)
re_whitespace = re.compile(r"\s+", flags=re.UNICODE)
re_citation = re.compile(r"\[\d+\]+(\s|\u200b|$)", flags=re.UNICODE)
re_text = re.compile(r"[\w\s]+", flags=re.UNICODE | re.IGNORECASE)
re_doi = re.compile(r"""\b(10[.][0-9]{3,}(?:[.][0-9]+)*/(?:(?!["&\'<>])\S)+)\b""", flags=re.UNICODE)
re_issn = re.compile(r"e?ISSN \d{4}\-\d{3}[X\d]", flags=re.UNICODE | re.IGNORECASE)
re_isbn = re.compile(r"ISBN (?:-?\d){10,13}", flags=re.UNICODE | re.IGNORECASE)


def main(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="fetch snippets of language-specific text from random pages "
                    "of one or many wikipedias",
    )
    parser.add_argument(
        "--langs", type=str, nargs="+", required=False,
        help="one or more languages for which to fetch wiki snippets "
             "as ISO 639-1 language codes",
    )
    parser.add_argument(
        "--langfile", type=str, required=False,
        help="path to text file on disk containing languages for which to fetch "
             "wiki snippets as ISO 639-1 language codes, one per line",
    )
    parser.add_argument(
        "--outdir", type=str, required=False,
        help="path to directory on disk to which wiki snippets will be saved",
    )
    parser.add_argument(
        "--n_snippets", "-n", type=int, default=25,
        help="number of text snippets to save per language",
    )
    args = parser.parse_args(argv)

    if not bool(args.langs) ^ bool(args.langfile):
        raise ValueError("either `langs` or `langfile` must be specified")
    if args.langfile:
        with io.open(args.langfile, mode="rt", encoding="utf-8") as f:
            langs = [line.strip() for line in f]
    else:
        langs = args.langs

    if args.outdir:
        outdir = os.path.realpath(os.path.expanduser(args.outdir))
    else:
        outdir = None

    wikipedia.set_rate_limiting(True, min_wait=datetime.timedelta(microseconds=20000))

    for i, lang in enumerate(langs):
        logging.info("lang: %s (%s/%s) ...", lang, i + 1, len(langs))
        snippets = get_snippets(lang, args.n_snippets)
        if outdir:
            fname = os.path.join(outdir, lang + ".txt")
            with io.open(fname, mode="wt", encoding="utf-8") as f:
                for snippet in snippets:
                    f.write(snippet + "\n")
            logging.info(
                "saved %s %s snippets to %s", args.n_snippets, lang, fname)
        else:
            logging.info(
                "fetched %s %s snippets but did not save to disk", args.n_snippets, lang)


def get_snippets(lang, n, timeout=3600):
    """
    Args:
        lang (str)
        n (int)
        timeout (int)

    Returns:
        List[str]
    """
    wikipedia.set_lang(lang)
    all_snippets = []
    seen_pages = set()
    start_time = time.time()
    with tqdm(total=n, unit="snippets", unit_scale=True, leave=True) as pbar:
        while len(all_snippets) < n and (time.time() - start_time) < timeout:
            for title in wikipedia.random(min(25, n)):
                if title in seen_pages or "/" in title:
                    continue
                try:
                    page = wikipedia.page(title=title)
                except Exception:
                    logging.debug("unable to fetch wiki page '%s'", title)
                    continue
                try:
                    snippets = extract_snippets_from_page(page, exclude_en=lang != "en")
                except Exception:
                    logging.exception("unable to extract snippets from page '%s'", title)
                    continue
                all_snippets.extend(snippets)
                pbar.update(len(snippets))
                # break out of random title loop early if we can
                if len(all_snippets) >= n:
                    break
    return all_snippets[:n]


def extract_snippets_from_page(
    page,
    min_content_len=1000,
    snippet_len=(200, 500),
    min_text_frac=0.9,
    exclude_en=False,
    max_n=3,
):
    """
    Args:
        page (:class:`wikipedia.WikipediaPage`)
        min_content_len (int)
        snippet_len (Tuple[int, int])
        min_text_frac (float)
        exclude_en (bool)
        max_n (int)

    Returns:
        List[str]
    """
    snippets = []
    content = page.content
    if len(content) < min_content_len:
        return snippets
    for section in re_sections.split(content):
        if len(snippets) >= max_n:
            break
        for snippet in re_breaking_whitespace.split(section.strip()):
            if len(snippets) >= max_n:
                break
            snippet = re_whitespace.sub(" ", snippet).strip()
            snippet = re_citation.sub("", snippet)
            if is_good_snippet(snippet, snippet_len, min_text_frac, exclude_en):
                snippets.append(snippet)
    return snippets


def is_good_snippet(snippet, len_range, min_text_frac, exclude_en):
    """
    Args:
        snippet (str)
        len_range (Tuple[int, int])
        min_text_frac (float)
        exclude_en (bool)

    Returns:
        bool
    """
    len_snippet = len(snippet)
    if len_snippet < len_range[0] or len_snippet >= len_range[1]:
        return False
    # make sure snippet is *mostly* text
    len_text = sum(match.end() - match.start() for match in re_text.finditer(snippet))
    if len_text / len_snippet < min_text_frac:
        return False
    # ugh, math and urls!
    if any(s in snippet for s in (r"\displaystyle", "http://", "https://")):
        return False
    # check for citations/references
    if any(re_pat.search(snippet) for re_pat in (re_doi, re_issn, re_isbn)):
        return False
    # filter out english copy-paste jobs
    if exclude_en is True:
        is_reliable, _, best_guesses = cld2_detect(snippet.encode("utf-8"), bestEffort=True)
        if is_reliable is True and best_guesses[0][1] == "en":
            logging.debug(
                "found english-heavy snippet in non-english wiki text:\n%s", snippet)
            return False
    return True


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))