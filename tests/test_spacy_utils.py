from __future__ import absolute_import, unicode_literals

import unittest

from textacy import cache, spacy_utils


class SpacyUtilsTestCase(unittest.TestCase):

    def setUp(self):
        spacy_lang = cache.load_spacy('en')
        text = """
        The unit tests aren't going well.
        I love Python, but I don't love backwards incompatibilities.
        No programmers were permanently damaged for textacy's sake.
        Thank God for Stack Overflow."""
        self.spacy_doc = spacy_lang(text.strip())

    def test_is_plural_noun(self):
        plural_nouns = [
            0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.assertEqual(
            [int(spacy_utils.is_plural_noun(tok)) for tok in self.spacy_doc],
            plural_nouns)

    def test_is_negated_verb(self):
        negated_verbs = [
            0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.assertEqual(
            [int(spacy_utils.is_negated_verb(tok)) for tok in self.spacy_doc],
            negated_verbs)

    def test_preserve_case(self):
        preserved_cases = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0]
        self.assertEqual(
            [int(spacy_utils.preserve_case(tok)) for tok in self.spacy_doc],
            preserved_cases)

    def test_normalize_str(self):
        normalized_strs = [
            'the', 'unit', 'test', 'be', 'not', 'go', 'well', '.', '-PRON-',
            'love', 'Python', ',', 'but', '-PRON-', 'do', 'not', 'love',
            'backwards', 'incompatibility', '.', 'no', 'programmer', 'be',
            'permanently', 'damage', 'for', 'textacy', "'s", 'sake', '.',
            'thank', 'God', 'for', 'Stack', 'Overflow', '.']
        self.assertEqual(
            [spacy_utils.normalized_str(tok) for tok in self.spacy_doc if not tok.is_space],
            normalized_strs)
