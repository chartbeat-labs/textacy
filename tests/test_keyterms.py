# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

import unittest

from textacy import data, keyterms, preprocess_text, spacy_utils


class ExtractTestCase(unittest.TestCase):
    """
    Note: Results of weighting nodes in a network by pagerank are random bc the
    algorithm relies on a random walk. Consequently, keyterm rankings aren't
    necessarily the same across runs.
    """

    def setUp(self):
        spacy_lang = data.load_spacy('en')
        text = """
        Friedman joined the London bureau of United Press International after completing his master's degree. He was dispatched a year later to Beirut, where he lived from June 1979 to May 1981 while covering the Lebanon Civil War. He was hired by The New York Times as a reporter in 1981 and re-dispatched to Beirut at the start of the 1982 Israeli invasion of Lebanon. His coverage of the war, particularly the Sabra and Shatila massacre, won him the Pulitzer Prize for International Reporting (shared with Loren Jenkins of The Washington Post). Alongside David K. Shipler he also won the George Polk Award for foreign reporting.

        In June 1984, Friedman was transferred to Jerusalem, where he served as the New York Times Jerusalem Bureau Chief until February 1988. That year he received a second Pulitzer Prize for International Reporting, which cited his coverage of the First Palestinian Intifada. He wrote a book, From Beirut to Jerusalem, describing his experiences in the Middle East, which won the 1989 U.S. National Book Award for Nonfiction.

        Friedman covered Secretary of State James Baker during the administration of President George H. W. Bush. Following the election of Bill Clinton in 1992, Friedman became the White House correspondent for the New York Times. In 1994, he began to write more about foreign policy and economics, and moved to the op-ed page of The New York Times the following year as a foreign affairs columnist. In 2002, Friedman won the Pulitzer Prize for Commentary for his "clarity of vision, based on extensive reporting, in commenting on the worldwide impact of the terrorist threat."

        In February 2002, Friedman met Saudi Crown Prince Abdullah and encouraged him to make a comprehensive attempt to end the Arab-Israeli conflict by normalizing Arab relations with Israel in exchange for the return of refugees alongside an end to the Israel territorial occupations. Abdullah proposed the Arab Peace Initiative at the Beirut Summit that March, which Friedman has since strongly supported.

        Friedman received the 2004 Overseas Press Club Award for lifetime achievement and was named to the Order of the British Empire by Queen Elizabeth II.

        In May 2011, The New York Times reported that President Barack Obama "has sounded out" Friedman concerning Middle East issues.
        """
        self.spacy_doc = spacy_lang(preprocess_text(text), parse=False)

    def test_sgrank(self):
        expected = [
            'new york times', 'york times jerusalem bureau chief', 'friedman',
            'president george h. w.', 'george polk award', 'pulitzer prize',
            'u.s. national book award', 'international reporting', 'beirut',
            'washington post']
        observed = [term for term, _ in keyterms.sgrank(self.spacy_doc)]
        self.assertEqual(len(expected), len(observed))
        # can't do this owing to randomness of results
        # for e, o in zip(expected, observed):
        #     self.assertEqual(e, o)

    def test_sgrank_ngrams_1(self):
        expected = [
            'friedman', 'international', 'beirut', 'bureau', 'york']
        observed = [term for term, _ in keyterms.sgrank(self.spacy_doc, ngrams=1, n_keyterms=5)]
        self.assertEqual(len(expected), len(observed))
        # can't do this owing to randomness of results
        # for e, o in zip(expected, observed):
        #     self.assertEqual(e, o)

    def test_sgrank_ngrams_1_2_3(self):
        expected = [
            'new york times', 'friedman', 'pulitzer prize', 'beirut',
            'international reporting']
        observed = [term for term, _ in keyterms.sgrank(self.spacy_doc, ngrams=(1, 2, 3), n_keyterms=5)]
        self.assertEqual(len(expected), len(observed))
        # can't do this owing to randomness of results
        # for e, o in zip(expected, observed):
        #     self.assertEqual(e, o)

    def test_sgrank_n_keyterms(self):
        expected = [
            'new york times', 'new york times jerusalem bureau chief', 'friedman',
            'president george h. w. bush', 'david k. shipler']
        observed = [
            term for term, _ in keyterms.sgrank(self.spacy_doc, n_keyterms=5)]
        self.assertEqual(len(expected), len(observed))
        # can't do this owing to randomness of results
        # for e, o in zip(expected, observed):
        #     self.assertEqual(e, o)

    def test_sgrank_norm_lower(self):
        expected = [
            'new york times', 'president george h. w. bush', 'friedman',
            'new york times jerusalem bureau', 'george polk award']
        observed = [
            term for term, _
            in keyterms.sgrank(self.spacy_doc, normalize='lower', n_keyterms=5)]
        self.assertEqual(len(expected), len(observed))
        for term in observed:
            self.assertEqual(term, term.lower())
        # can't do this owing to randomness of results
        # for e, o in zip(expected, observed):
        #     self.assertEqual(e, o)

    def test_sgrank_norm_none(self):
        expected = [
            'New York Times', 'New York Times Jerusalem Bureau Chief', 'Friedman',
            'President George H. W. Bush', 'George Polk Award']
        observed = [
            term for term, _
            in keyterms.sgrank(self.spacy_doc, normalize=None, n_keyterms=5)]
        self.assertEqual(len(expected), len(observed))
        # can't do this owing to randomness of results
        # for e, o in zip(expected, observed):
        #     self.assertEqual(e, o)

    def test_sgrank_norm_normalized_str(self):
        expected = [
            'New York Times', 'New York Times Jerusalem Bureau Chief', 'Friedman',
            'President George H. W. Bush', 'George Polk Award']
        observed = [
            term for term, _
            in keyterms.sgrank(self.spacy_doc, normalize=spacy_utils.normalized_str, n_keyterms=5)]
        self.assertEqual(len(expected), len(observed))
        # can't do this owing to randomness of results
        # for e, o in zip(expected, observed):
        #     self.assertEqual(e, o)

    def test_sgrank_window_width(self):
        expected = [
            'new york times', 'friedman', 'new york times jerusalem',
            'times jerusalem bureau', 'second pulitzer prize']
        observed = [
            term for term, _
            in keyterms.sgrank(self.spacy_doc, window_width=50, n_keyterms=5)]
        self.assertEqual(len(expected), len(observed))
        # can't do this owing to randomness of results
        # for e, o in zip(expected, observed):
        #     self.assertEqual(e, o)

    def test_textrank(self):
        expected = [
            'friedman', 'beirut', 'reporting', 'arab', 'new', 'award', 'foreign',
            'year', 'times', 'jerusalem']
        observed = [
            term for term, _ in keyterms.textrank(self.spacy_doc)]
        self.assertEqual(len(expected), len(observed))
        # can't do this owing to randomness of results
        # for e, o in zip(expected, observed):
        #     self.assertEqual(e, o)

    def test_textrank_n_keyterms(self):
        expected = ['friedman', 'beirut', 'reporting', 'arab', 'new']
        observed = [
            term for term, _ in keyterms.textrank(self.spacy_doc, n_keyterms=5)]
        self.assertEqual(len(expected), len(observed))
        # can't do this owing to randomness of results
        # for e, o in zip(expected, observed):
        #     self.assertEqual(e, o)

    def test_textrank_norm_lower(self):
        expected = ['friedman', 'beirut', 'reporting', 'arab', 'new']
        observed = [
            term for term, _
            in keyterms.textrank(self.spacy_doc, normalize='lower', n_keyterms=5)]
        self.assertEqual(len(expected), len(observed))
        # can't do this owing to randomness of results
        # for e, o in zip(expected, observed):
        #     self.assertEqual(e, o)
        for term in observed:
            self.assertEqual(term, term.lower())

    def test_textrank_norm_none(self):
        expected = ['Friedman', 'Beirut', 'New', 'Arab', 'Award']
        observed = [
            term for term, _
            in keyterms.textrank(self.spacy_doc, normalize=None, n_keyterms=5)]
        self.assertEqual(len(expected), len(observed))
        # can't do this owing to randomness of results
        # for e, o in zip(expected, observed):
        #     self.assertEqual(e, o)

    def test_textrank_norm_normalized_str(self):
        expected = ['Friedman', 'Beirut', 'New', 'Award', 'foreign']
        observed = [
            term for term, _
            in keyterms.textrank(self.spacy_doc, normalize=spacy_utils.normalized_str, n_keyterms=5)]
        self.assertEqual(len(expected), len(observed))
        # can't do this owing to randomness of results
        # for e, o in zip(expected, observed):
        #     self.assertEqual(e, o)

    def test_singlegrank(self):
        expected = [
            'new york times jerusalem bureau', 'new york times', 'friedman',
            'foreign reporting', 'international reporting', 'pulitzer prize',
            'book award', 'press international', 'president george', 'beirut']
        observed = [term for term, _ in keyterms.singlerank(self.spacy_doc)]
        self.assertEqual(len(expected), len(observed))
        # can't do this owing to randomness of results
        # for e, o in zip(expected, observed):
        #     self.assertEqual(e, o)

    def test_singlegrank_n_keyterms(self):
        expected = [
            'new york times jerusalem bureau', 'new york times', 'friedman',
            'foreign reporting', 'international reporting']
        observed = [
            term for term, _ in keyterms.singlerank(self.spacy_doc, n_keyterms=5)]
        self.assertEqual(len(expected), len(observed))
        # can't do this owing to randomness of results
        # for e, o in zip(expected, observed):
        #     self.assertEqual(e, o)

    def test_singlegrank_norm_lower(self):
        expected = [
            'new york times jerusalem bureau', 'new york times', 'friedman',
            'foreign reporting', 'international reporting']
        observed = [
            term for term, _
            in keyterms.singlerank(self.spacy_doc, normalize='lower', n_keyterms=5)]
        self.assertEqual(len(expected), len(observed))
        # can't do this owing to randomness of results
        # for e, o in zip(expected, observed):
        #     self.assertEqual(e, o)
        for term in observed:
            self.assertEqual(term, term.lower())

    def test_singlegrank_norm_none(self):
        expected = [
            'New York Times Jerusalem', 'New York Times', 'Friedman',
            'Pulitzer Prize', 'foreign reporting']
        observed = [
            term for term, _
            in keyterms.singlerank(self.spacy_doc, normalize=None, n_keyterms=5)]
        self.assertEqual(len(expected), len(observed))
        # can't do this owing to randomness of results
        # for e, o in zip(expected, observed):
        #     self.assertEqual(e, o)

    def test_singlegrank_norm_normalized_str(self):
        expected = [
            'New York Times Jerusalem', 'New York Times', 'Friedman',
            'Pulitzer Prize', 'foreign reporting']
        observed = [
            term for term, _
            in keyterms.singlerank(self.spacy_doc, normalize=spacy_utils.normalized_str, n_keyterms=5)]
        self.assertEqual(len(expected), len(observed))
        # can't do this owing to randomness of results
        # for e, o in zip(expected, observed):
        #     self.assertEqual(e, o)
