# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

from textacy import text_utils

GOOD_ACRONYMS = [
    "LGTM",
    "U.S.A.",
    "PEP8",
    "LGBTQQI2S",
    "TF-IDF",
    "D3",
    "3D",
    "3-D",
    "3D-TV",
    "D&D",
    "PrEP",
    "H2SO4",
    "I/O",
    "WASPs",
    "G-8",
    "A-TReC",
]
BAD_ACRONYMS = [
    "A",
    "GHz",
    "1a",
    "D o E",
    "Ms",
    "Ph.D",
    "3-Dim.",
    "the",
    "FooBar",
    "1",
    " ",
    "",
]

CRUFTY_TERMS = [
    "( foo bar )",
    "foo -bar",
    "- 123.4",
    ".-foo bar",
    "?!foo",
    "bar?!",
    "foo 's bar",
    "foo 'll bar",
    "  foo   bar   ",
    "foo bar.   ",
]
GOOD_TERMS = ["foo (bar)", "foo?", "bar!", "-123.4"]
BAD_TERMS = ["(foo bar", "foo) bar", "?>,!-.", "", "foo) (bar"]

LANG_SENTS = [
    ("en", "This sentence is in English."),
    ("es", "Esta oración es en Español."),
    ("fr", "Cette phrase est en français."),
    ("un", "1"),
    ("un", " "),
    ("un", ""),
]

TEXT = """
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


def test_is_acronym_good():
    for item in GOOD_ACRONYMS:
        assert text_utils.is_acronym(item)


def test_is_acronym_bad():
    for item in BAD_ACRONYMS:
        assert not text_utils.is_acronym(item)


def test_is_acronym_exclude():
    assert not text_utils.is_acronym("NASA", exclude={"NASA"})


def test_detect_language():
    for lang, sent in LANG_SENTS:
        assert text_utils.detect_language(sent) == lang


def test_keyword_in_context_keyword():
    for keyword in ("clinton", "all"):
        results = list(
            text_utils.keyword_in_context(
                TEXT, keyword, ignore_case=True, window_width=50, print_only=False
            )
        )
        for pre, kw, post in results:
            assert kw.lower() == keyword


def test_keyword_in_context_ignore_case():
    for keyword in ("All", "all"):
        results = list(
            text_utils.keyword_in_context(
                TEXT, keyword, ignore_case=False, window_width=50, print_only=False
            )
        )
        for pre, kw, post in results:
            assert kw == keyword
    # also test for a null result, bc of case
    results = list(
        text_utils.keyword_in_context(
            TEXT, "clinton", ignore_case=False, window_width=50, print_only=False
        )
    )
    assert results == []


def test_keyword_in_context_window_width():
    for window_width in (10, 20):
        results = list(
            text_utils.keyword_in_context(
                TEXT,
                "clinton",
                ignore_case=True,
                print_only=False,
                window_width=window_width,
            )
        )
        for pre, kw, post in results:
            assert len(pre) <= window_width
            assert len(post) <= window_width


def test_keyword_in_context_unicode():
    keyword = "terminó"
    results = list(
        text_utils.keyword_in_context(
            "No llores porque ya se terminó, sonríe porque sucedió.",
            keyword,
            print_only=False,
        )
    )
    for pre, kw, post in results:
        assert kw == keyword


def test_clean_terms_good():
    observed = list(text_utils.clean_terms(GOOD_TERMS))
    assert observed == GOOD_TERMS


def test_clean_terms_bad():
    observed = list(text_utils.clean_terms(BAD_TERMS))
    assert observed == []


def test_clean_terms_crufty():
    observed = list(text_utils.clean_terms(CRUFTY_TERMS))
    expected = [
        "(foo bar)",
        "foo-bar",
        "-123.4",
        "foo bar",
        "foo",
        "bar?!",
        "foo's bar",
        "foo'll bar",
        "foo bar",
        "foo bar.",
    ]
    assert observed == expected
