# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

import unittest

from textacy import preprocess


class PreprocessTestCase(unittest.TestCase):

    def test_normalize_whitespace(self):
        text = "Hello, world!  Hello...\t \tworld?\n\nHello:\r\n\n\nWorld. "
        proc_text = "Hello, world! Hello... world?\nHello:\nWorld."
        self.assertEqual(preprocess.normalize_whitespace(text), proc_text)

    def test_unpack_contractions(self):
        text = "Y'all can't believe you're not who they've said I'll become, but shouldn't."
        proc_text = "You all can not believe you are not who they have said I will become, but should not."
        self.assertEqual(preprocess.unpack_contractions(text), proc_text)

    def test_replace_urls(self):
        text = "I learned everything I know from www.stackoverflow.com and http://wikipedia.org/ and Mom."
        proc_text = "I learned everything I know from *URL* and *URL* and Mom."
        self.assertEqual(preprocess.replace_urls(text, '*URL*'), proc_text)

    def test_replace_emails(self):
        text = "I can be reached at username@example.com through next Friday."
        proc_text = "I can be reached at *EMAIL* through next Friday."
        self.assertEqual(preprocess.replace_emails(text, '*EMAIL*'), proc_text)

    def test_replace_phone_numbers(self):
        text = "I can be reached at 555-123-4567 through next Friday."
        proc_text = "I can be reached at *PHONE* through next Friday."
        self.assertEqual(preprocess.replace_phone_numbers(text, '*PHONE*'), proc_text)

    def test_replace_numbers(self):
        text = "I owe $1,000.99 to 123 people for 2 +1 reasons."
        proc_text = "I owe $*NUM* to *NUM* people for *NUM* *NUM* reasons."
        self.assertEqual(preprocess.replace_numbers(text, '*NUM*'), proc_text)

    def test_remove_punct(self):
        text = "I can't. No, I won't! It's a matter of \"principle\"; of -- what's the word? -- conscience."
        proc_text = "I cant No I wont Its a matter of principle of  whats the word  conscience"
        self.assertEqual(preprocess.remove_punct(text), proc_text)

    def test_remove_punct_marks(self):
        text = "I can't. No, I won't! It's a matter of \"principle\"; of -- what's the word? -- conscience."
        proc_text = "I cant. No, I wont! Its a matter of principle; of  whats the word?  conscience."
        self.assertEqual(preprocess.remove_punct(text, marks="-'\""), proc_text)

    def test_replace_currency_symbols(self):
        tests = [
            ('$1.00 equals £0.67 equals €0.91.',
             'USD1.00 equals GBP0.67 equals EUR0.91.',
             '*CUR* 1.00 equals *CUR* 0.67 equals *CUR* 0.91.'),
            ('this zebra costs $100.',
             'this zebra costs USD100.',
             'this zebra costs *CUR* 100.'),
            ]
        for text, proc_text1, proc_text2 in tests:
            self.assertEqual(preprocess.replace_currency_symbols(text, replace_with=None), proc_text1)
            self.assertEqual(preprocess.replace_currency_symbols(text, replace_with='*CUR* '), proc_text2)

    def test_remove_accents(self):
        text = "El niño se asustó -- qué miedo!"
        proc_text = "El nino se asusto -- que miedo!"
        self.assertEqual(preprocess.remove_accents(text, method='unicode'), proc_text)
        self.assertEqual(preprocess.remove_accents(text, method='ascii'), proc_text)
        self.assertRaises(Exception, preprocess.remove_accents, text, method='foo')
