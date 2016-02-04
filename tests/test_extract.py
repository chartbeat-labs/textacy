from __future__ import absolute_import, unicode_literals

import re
import unittest

from textacy import data, extract, preprocess, regexes_etc


class ExtractTestCase(unittest.TestCase):

    def setUp(self):
        self.maxDiff = None
        spacy_pipeline = data.load_spacy_pipeline(lang='en')
        text = """
            The hedge fund magnates Daniel S. Loeb, Louis Moore Bacon and Steven A. Cohen have much in common. They have managed billions of dollars in capital, earning vast fortunes. They have invested millions in art — and millions more in political candidates.
            Moreover, each has exploited an esoteric tax loophole that saved them millions in taxes. The trick? Route the money to Bermuda and back.
            With inequality at its highest levels in nearly a century and public debate rising over whether the government should respond to it through higher taxes on the wealthy, the very richest Americans have financed a sophisticated and astonishingly effective apparatus for shielding their fortunes. Some call it the "income defense industry," consisting of a high-priced phalanx of lawyers, estate planners, lobbyists and anti-tax activists who exploit and defend a dizzying array of tax maneuvers, virtually none of them available to taxpayers of more modest means.
            In recent years, this apparatus has become one of the most powerful avenues of influence for wealthy Americans of all political stripes, including Mr. Loeb and Mr. Cohen, who give heavily to Republicans, and the liberal billionaire George Soros, who has called for higher levies on the rich while at the same time using tax loopholes to bolster his own fortune.
            All are among a small group providing much of the early cash for the 2016 presidential campaign.
            Operating largely out of public view — in tax court, through arcane legislative provisions, and in private negotiations with the Internal Revenue Service — the wealthy have used their influence to steadily whittle away at the government's ability to tax them. The effect has been to create a kind of private tax system, catering to only several thousand Americans.
            The impact on their own fortunes has been stark. Two decades ago, when Bill Clinton was elected president, the 400 highest-earning taxpayers in America paid nearly 27 percent of their income in federal taxes, according to I.R.S. data. By 2012, when President Obama was re-elected, that figure had fallen to less than 17 percent, which is just slightly more than the typical family making $100,000 annually, when payroll taxes are included for both groups.
            The ultra-wealthy "literally pay millions of dollars for these services," said Jeffrey A. Winters, a political scientist at Northwestern University who studies economic elites, "and save in the tens or hundreds of millions in taxes."
            Some of the biggest current tax battles are being waged by some of the most generous supporters of 2016 candidates. They include the families of the hedge fund investors Robert Mercer, who gives to Republicans, and James Simons, who gives to Democrats; as well as the options trader Jeffrey Yass, a libertarian-leaning donor to Republicans.
            Mr. Yass's firm is litigating what the agency deemed to be tens of millions of dollars in underpaid taxes. Renaissance Technologies, the hedge fund Mr. Simons founded and which Mr. Mercer helps run, is currently under review by the I.R.S. over a loophole that saved their fund an estimated $6.8 billion in taxes over roughly a decade, according to a Senate investigation. Some of these same families have also contributed hundreds of thousands of dollars to conservative groups that have attacked virtually any effort to raises taxes on the wealthy.
            In the heat of the presidential race, the influence of wealthy donors is being tested. At stake are the Obama administration's limited 2013 tax increase on high earners — the first in two decades — and an I.R.S. initiative to ensure that, in effect, the higher rate sticks by cracking down on tax avoidance by the wealthy.
            While Democrats like Bernie Sanders and Hillary Clinton have pledged to raise taxes on these voters, virtually every Republican has advanced policies that would vastly reduce their tax bills, sometimes to as little as 10 percent of their income.
            At the same time, most Republican candidates favor eliminating the inheritance tax, a move that would allow the new rich, and the old, to bequeath their fortunes intact, solidifying the wealth gap far into the future. And several have proposed a substantial reduction — or even elimination — in the already deeply discounted tax rates on investment gains, a foundation of the most lucrative tax strategies.
            "There's this notion that the wealthy use their money to buy politicians; more accurately, it's that they can buy policy, and specifically, tax policy," said Jared Bernstein, a senior fellow at the left-leaning Center on Budget and Policy Priorities who served as chief economic adviser to Vice President Joseph R. Biden Jr. "That's why these egregious loopholes exist, and why it's so hard to close them."
            """
        # self.spacy_doc = spacy_pipeline(preprocess.normalize_whitespace(text))
        self.spacy_doc = spacy_pipeline(re.sub(r'\s+', ' ', text))

    def test_words(self):
        expected = [
            'The', 'hedge', 'fund', 'magnates', 'Daniel', 'S.', 'Loeb', ',',
            'Louis', 'Moore', 'Bacon', 'and', 'Steven', 'A.', 'Cohen', 'have',
            'much', 'in', 'common', '.', 'They', 'have', 'managed', 'billions',
            'of']
        observed = [tok.orth_ for tok in extract.words(
            self.spacy_doc, filter_stops=False, filter_punct=False, filter_nums=False)[:25]]
        self.assertEqual(observed, expected)

    def test_words_filter(self):
        expected = [
            'hedge', 'fund', 'magnates', 'Daniel', 'S.', 'Loeb', 'Louis',
            'Moore', 'Bacon', 'Steven', 'A.', 'Cohen', 'common', 'managed',
            'billions', 'dollars', 'capital', 'earning', 'vast', 'fortunes',
            'invested', 'millions', 'art', 'millions', 'political']
        observed = [tok.orth_ for tok in extract.words(
            self.spacy_doc, filter_stops=True, filter_punct=True, filter_nums=True)[:25]]
        self.assertEqual(observed, expected)

    def test_words_good_tags(self):
        expected = [
            'hedge', 'fund', 'Daniel', 'S.', 'Loeb', 'Louis', 'Moore', 'Bacon',
            'Steven', 'A.', 'Cohen', 'They', 'billions', 'dollars', 'capital',
            'fortunes', 'They', 'millions', 'art', 'millions', 'candidates',
            'tax', 'loophole', 'them', 'millions']
        observed = [tok.orth_ for tok in extract.words(
            self.spacy_doc, filter_stops=False, filter_punct=False, filter_nums=False,
            good_pos_tags={'NOUN'})[:25]]
        self.assertEqual(observed, expected)

    def test_words_min_freq(self):
        expected = [
            'The', 'hedge', 'fund', ',', 'and', 'have', 'in', '.', 'They',
            'have', 'of', 'dollars', 'in', ',', 'fortunes', '.', 'They', 'have',
            'millions', 'in', '—', 'and', 'millions', 'more', 'in']
        observed = [tok.orth_ for tok in extract.words(
            self.spacy_doc, filter_stops=False, filter_punct=False, filter_nums=False,
            min_freq=3)[:25]]
        self.assertEqual(observed, expected)

    def test_ngrams_less_than_1(self):
        self.assertRaises(ValueError, extract.ngrams, self.spacy_doc, 0)

    def test_ngrams_1(self):
        expected = [
            'The', 'hedge', 'fund', 'magnates', 'Daniel', 'S.', 'Loeb', ',',
            'Louis', 'Moore', 'Bacon', 'and', 'Steven', 'A.', 'Cohen', 'have',
            'much', 'in', 'common', '.', 'They', 'have', 'managed', 'billions', 'of']
        observed = [span.orth_ for span in extract.ngrams(
            self.spacy_doc, 1, filter_stops=False, filter_punct=False, filter_nums=False)[:25]]
        self.assertEqual(observed, expected)

    def test_ngrams_2(self):
        expected = [
            'The hedge', 'hedge fund', 'fund magnates', 'magnates Daniel', 'Daniel S.',
            'S. Loeb', 'Loeb,', ', Louis', 'Louis Moore', 'Moore Bacon', 'Bacon and',
            'and Steven', 'Steven A.', 'A. Cohen', 'Cohen have', 'have much', 'much in',
            'in common', 'common.', '. They', 'They have', 'have managed',
            'managed billions', 'billions of', 'of dollars']
        observed = [span.orth_ for span in extract.ngrams(
            self.spacy_doc, 2, filter_stops=False, filter_punct=False, filter_nums=False)[:25]]
        self.assertEqual(observed, expected)

    def test_ngrams_filter(self):
        expected = [
            'hedge fund', 'fund magnates', 'magnates Daniel', 'Daniel S.', 'S. Loeb',
            'Louis Moore', 'Moore Bacon', 'Steven A.', 'A. Cohen', 'managed billions',
            'earning vast', 'vast fortunes', 'invested millions', 'political candidates',
            'esoteric tax', 'tax loophole', 'highest levels', 'public debate',
            'debate rising', 'higher taxes', 'richest Americans', 'astonishingly effective',
            'effective apparatus', 'income defense', 'defense industry']
        observed = [span.orth_ for span in extract.ngrams(
            self.spacy_doc, 2, filter_stops=True, filter_punct=True, filter_nums=True)[:25]]
        self.assertEqual(observed, expected)

    def test_ngrams_min_freq(self):
        expected = [
            'The hedge', 'hedge fund', '. They', 'of dollars', 'fortunes.', '. They',
            'millions in', 'millions in', 'in taxes', 'taxes.', '. The', 'taxes on',
            'on the', 'the wealthy', ', the', 'fortunes.', ',"', 'of the', 'the most',
            ', who', 'who give', 'to Republicans', ', and', ', who', 'on the']
        observed = [span.orth_ for span in extract.ngrams(
            self.spacy_doc, 2, filter_stops=False, filter_punct=False, filter_nums=False,
            min_freq=3)[:25]]
        self.assertEqual(observed, expected)

    def test_ngrams_good_tag(self):
        expected = [
            'hedge fund', 'Daniel S.', 'S. Loeb', 'Louis Moore', 'Moore Bacon',
            'Steven A.', 'A. Cohen', 'tax loophole', 'them millions', 'income defense',
            'defense industry', 'estate planners', 'tax activists', 'activists who',
            'tax maneuvers', 'Mr. Loeb', 'Mr. Cohen', 'billionaire George', 'George Soros',
            'tax loopholes', 'tax court', 'Internal Revenue', 'Revenue Service',
            'tax system', 'Bill Clinton']
        observed = [span.orth_ for span in extract.ngrams(
            self.spacy_doc, 2, filter_stops=False, filter_punct=False, filter_nums=False,
            good_pos_tags={'NOUN'})[:25]]
        self.assertEqual(observed, expected)

    def test_named_entities(self):
        expected = [
            'Daniel S. Loeb', 'Louis Moore Bacon', 'Steven A. Cohen', 'billions of dollars',
            'millions', 'millions', 'millions', 'Bermuda', 'nearly a century', 'Americans',
            'recent years', 'Americans', 'Loeb', 'Cohen', 'Republicans', 'George Soros',
            '2016', 'Internal Revenue Service', 'several thousand', 'Americans',
            'Two decades ago', 'Bill Clinton', '400', 'America', 'nearly 27 percent']
        observed = [ent.text for ent in extract.named_entities(
            self.spacy_doc, drop_determiners=True)][:25]
        self.assertEqual(observed, expected)

    def test_named_entities_good(self):
        expected = [
            'Daniel S. Loeb', 'Louis Moore Bacon', 'Steven A. Cohen', 'Loeb', 'Cohen',
            'George Soros', 'Bill Clinton', 'Obama', 'Jeffrey A. Winters', 'Robert Mercer',
            'James Simons', 'Jeffrey Yass', 'Yass', 'Simons', 'Mercer', 'Hillary Clinton',
            'Jared Bernstein', 'Joseph R. Biden Jr.']
        observed = [ent.text for ent in extract.named_entities(
            self.spacy_doc, good_ne_types={'PERSON'}, drop_determiners=True)][:25]
        self.assertEqual(observed, expected)

    def test_named_entities_min_freq(self):
        expected = [
            'millions', 'millions', 'millions', 'Americans', 'Americans', 'Republicans',
            '2016', 'Americans', 'Obama', '2016', 'Republicans', 'Democrats', 'Republicans',
            'Obama', 'Democrats', 'Republican', 'Republican']
        observed = [ent.text for ent in extract.named_entities(
            self.spacy_doc, drop_determiners=True, min_freq=2)]
        self.assertEqual(observed, expected)

    def test_named_entities_determiner(self):
        expected = ['the Internal Revenue Service', 'an estimated $6.8 billion']
        observed = [ent.text for ent in extract.named_entities(
            self.spacy_doc, drop_determiners=False) if ent[0].pos_ == 'DET']
        self.assertEqual(observed, expected)

    def test_noun_phrases(self):
        expected = [
            'hedge fund', 'Daniel S. Loeb', 'Steven A. Cohen', 'They', 'billions',
            'dollars', 'capital', 'vast fortunes', 'They', 'millions', 'art', 'millions',
            'political candidates', 'esoteric tax loophole', 'them', 'millions', 'taxes',
            'money', 'Bermuda', 'inequality', 'its highest levels', 'nearly a century',
            'public debate', 'government', 'it']
        observed = [np.text for np in extract.noun_phrases(
            self.spacy_doc, drop_determiners=True)][:25]
        self.assertEqual(observed, expected)

    def test_noun_phrases_determiner(self):
        expected = [
            'The hedge fund', 'Daniel S. Loeb', 'Steven A. Cohen', 'They', 'billions',
            'dollars', 'capital', 'vast fortunes', 'They', 'millions', 'art', 'millions',
            'political candidates', 'an esoteric tax loophole', 'them', 'millions',
            'taxes', 'the money', 'Bermuda', 'inequality', 'its highest levels',
            'nearly a century', 'public debate', 'the government', 'it']
        observed = [np.text for np in extract.noun_phrases(
            self.spacy_doc, drop_determiners=False)][:25]
        self.assertEqual(observed, expected)

    def test_noun_phrases_min_freq(self):
        expected = [
            'hedge fund', 'They', 'dollars', 'They', 'millions', 'millions', 'them',
            'millions', 'taxes', 'it', 'their fortunes', 'it', 'who', 'them', 'influence',
            'who', 'Republicans', 'who', 'same time', 'them', 'effect', 'their income',
            'millions', 'dollars', 'who']
        observed = [np.text for np in extract.noun_phrases(
            self.spacy_doc, drop_determiners=True, min_freq=2)][:25]
        self.assertEqual(observed, expected)

    def test_pos_regex_matches(self):
        expected = [
            'The hedge fund', 'Daniel S. Loeb', 'Louis Moore Bacon', 'Steven A. Cohen',
            'common. They', 'billions', 'dollars', 'capital', 'vast fortunes', 'They',
            'millions', 'art', 'millions', 'political candidates', 'an esoteric tax loophole',
            'them millions', 'taxes', 'The trick', 'Route', 'the money', 'Bermuda',
            'inequality', 'its highest levels', 'a century', 'public debate']
        observed = [span.text for span in extract.pos_regex_matches(
            self.spacy_doc, regexes_etc.POS_REGEX_PATTERNS['en']['NP'])][:25]
        self.assertEqual(observed, expected)

    def test_subject_verb_object_triples(self):
        expected = [
            'They, have managed, billions', 'They, have invested, millions',
            'each, has exploited, tax loophole', 'that, saved, them', 'that, saved, millions',
            'Americans, have financed, apparatus', 'Some, call, it', 'Some, call, income defense industry',
            'Some, call, consisting', 'apparatus, has become, one', 'who, give, to',
            'wealthy, have used, influence', 'wealthy, have used, whittle',
            'Bill Clinton, was elected, president', 'taxpayers, paid, percent',
            'family, making, 100,000', 'wealthy, pay, millions', 'who, studies, elites',
            'They, include, families', 'who, gives, to', 'who, gives, to',
            'agency, deemed, to be', 'that, saved, fund', 'that, saved, billion',
            'Some, contributed, thousands']
        observed = [', '.join(item.text for item in triple) for triple in
                    extract.subject_verb_object_triples(self.spacy_doc)[:25]]
        self.assertEqual(observed, expected)

    def test_acronyms_and_definitions(self):
        expected = {'I.R.S.': ''}
        observed = extract.acronyms_and_definitions(self.spacy_doc)
        self.assertEqual(observed, expected)
