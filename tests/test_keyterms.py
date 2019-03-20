# -*- coding: utf-8 -*-
"""
Note: Results of weighting nodes in a network by pagerank are random bc the
algorithm relies on a random walk. Consequently, keyterm rankings aren't
necessarily the same across runs.
"""
from __future__ import absolute_import, unicode_literals

import pytest

from textacy import cache, keyterms, preprocess_text
from textacy.spacier import utils as spacy_utils


@pytest.fixture(scope="module")
def spacy_doc():
    spacy_lang = cache.load_spacy("en")
    text = """
    Friedman joined the London bureau of United Press International after completing his master's degree. He was dispatched a year later to Beirut, where he lived from June 1979 to May 1981 while covering the Lebanon Civil War. He was hired by The New York Times as a reporter in 1981 and re-dispatched to Beirut at the start of the 1982 Israeli invasion of Lebanon. His coverage of the war, particularly the Sabra and Shatila massacre, won him the Pulitzer Prize for International Reporting (shared with Loren Jenkins of The Washington Post). Alongside David K. Shipler he also won the George Polk Award for foreign reporting.

    In June 1984, Friedman was transferred to Jerusalem, where he served as the New York Times Jerusalem Bureau Chief until February 1988. That year he received a second Pulitzer Prize for International Reporting, which cited his coverage of the First Palestinian Intifada. He wrote a book, From Beirut to Jerusalem, describing his experiences in the Middle East, which won the 1989 U.S. National Book Award for Nonfiction.

    Friedman covered Secretary of State James Baker during the administration of President George H. W. Bush. Following the election of Bill Clinton in 1992, Friedman became the White House correspondent for the New York Times. In 1994, he began to write more about foreign policy and economics, and moved to the op-ed page of The New York Times the following year as a foreign affairs columnist. In 2002, Friedman won the Pulitzer Prize for Commentary for his "clarity of vision, based on extensive reporting, in commenting on the worldwide impact of the terrorist threat."

    In February 2002, Friedman met Saudi Crown Prince Abdullah and encouraged him to make a comprehensive attempt to end the Arab-Israeli conflict by normalizing Arab relations with Israel in exchange for the return of refugees alongside an end to the Israel territorial occupations. Abdullah proposed the Arab Peace Initiative at the Beirut Summit that March, which Friedman has since strongly supported.

    Friedman received the 2004 Overseas Press Club Award for lifetime achievement and was named to the Order of the British Empire by Queen Elizabeth II.

    In May 2011, The New York Times reported that President Barack Obama "has sounded out" Friedman concerning Middle East issues.
    """
    spacy_doc = spacy_lang(preprocess_text(text), disable=["parser"])
    return spacy_doc


@pytest.fixture(scope="module")
def empty_spacy_doc():
    spacy_lang = cache.load_spacy("en")
    return spacy_lang("")


def test_sgrank(spacy_doc):
    expected = [
        "new york times",
        "york times jerusalem bureau chief",
        "friedman",
        "president george h. w.",
        "george polk award",
        "pulitzer prize",
        "u.s. national book award",
        "international reporting",
        "beirut",
        "washington post",
    ]
    observed = [term for term, _ in keyterms.sgrank(spacy_doc)]
    assert len(expected) == len(observed)
    # can't do this owing to randomness of results
    # for e, o in zip(expected, observed):
    #     assert e == o


def test_sgrank_ngrams_1(spacy_doc):
    expected = ["friedman", "international", "beirut", "bureau", "york"]
    observed = [term for term, _ in keyterms.sgrank(spacy_doc, ngrams=1, n_keyterms=5)]
    assert len(expected) == len(observed)
    # can't do this owing to randomness of results
    # for e, o in zip(expected, observed):
    #     assert e == o


def test_sgrank_ngrams_1_2_3(spacy_doc):
    expected = [
        "new york times",
        "friedman",
        "pulitzer prize",
        "beirut",
        "international reporting",
    ]
    observed = [
        term for term, _ in keyterms.sgrank(spacy_doc, ngrams=(1, 2, 3), n_keyterms=5)
    ]
    assert len(expected) == len(observed)
    # can't do this owing to randomness of results
    # for e, o in zip(expected, observed):
    #     asert e == o


def test_sgrank_n_keyterms(spacy_doc):
    expected = [
        "new york times",
        "new york times jerusalem bureau chief",
        "friedman",
        "president george h. w. bush",
        "david k. shipler",
    ]
    observed = [term for term, _ in keyterms.sgrank(spacy_doc, n_keyterms=5)]
    assert len(expected) == len(observed)
    # can't do this owing to randomness of results
    # for e, o in zip(expected, observed):
    #     asert e == o


def test_sgrank_norm_lower(spacy_doc):
    expected = [
        "new york times",
        "president george h. w. bush",
        "friedman",
        "new york times jerusalem bureau",
        "george polk award",
    ]
    observed = [
        term for term, _ in keyterms.sgrank(spacy_doc, normalize="lower", n_keyterms=5)
    ]
    assert len(expected) == len(observed)
    for term in observed:
        assert term == term.lower()
    # can't do this owing to randomness of results
    # for e, o in zip(expected, observed):
    #     asert e == o


def test_sgrank_norm_none(spacy_doc):
    expected = [
        "New York Times",
        "New York Times Jerusalem Bureau Chief",
        "Friedman",
        "President George H. W. Bush",
        "George Polk Award",
    ]
    observed = [
        term for term, _ in keyterms.sgrank(spacy_doc, normalize=None, n_keyterms=5)
    ]
    assert len(expected) == len(observed)
    # can't do this owing to randomness of results
    # for e, o in zip(expected, observed):
    #     asert e == o


def test_sgrank_norm_normalized_str(spacy_doc):
    expected = [
        "New York Times",
        "New York Times Jerusalem Bureau Chief",
        "Friedman",
        "President George H. W. Bush",
        "George Polk Award",
    ]
    observed = [
        term
        for term, _ in keyterms.sgrank(
            spacy_doc, normalize=spacy_utils.get_normalized_text, n_keyterms=5
        )
    ]
    assert len(expected) == len(observed)
    # can't do this owing to randomness of results
    # for e, o in zip(expected, observed):
    #     asert e == o


def test_sgrank_window_width(spacy_doc):
    expected = [
        "new york times",
        "friedman",
        "new york times jerusalem",
        "times jerusalem bureau",
        "second pulitzer prize",
    ]
    observed = [
        term for term, _ in keyterms.sgrank(spacy_doc, window_width=50, n_keyterms=5)
    ]
    assert len(expected) == len(observed)
    # can't do this owing to randomness of results
    # for e, o in zip(expected, observed):
    #     asert e == o


def test_textrank(spacy_doc):
    expected = [
        "friedman",
        "beirut",
        "reporting",
        "arab",
        "new",
        "award",
        "foreign",
        "year",
        "times",
        "jerusalem",
    ]
    observed = [term for term, _ in keyterms.textrank(spacy_doc)]
    assert len(expected) == len(observed)
    # can't do this owing to randomness of results
    # for e, o in zip(expected, observed):
    #     asert e == o


def test_textrank_n_keyterms(spacy_doc):
    expected = ["friedman", "beirut", "reporting", "arab", "new"]
    observed = [term for term, _ in keyterms.textrank(spacy_doc, n_keyterms=5)]
    assert len(expected) == len(observed)
    # can't do this owing to randomness of results
    # for e, o in zip(expected, observed):
    #     asert e == o


def test_textrank_norm_lower(spacy_doc):
    expected = ["friedman", "beirut", "reporting", "arab", "new"]
    observed = [
        term
        for term, _ in keyterms.textrank(spacy_doc, normalize="lower", n_keyterms=5)
    ]
    assert len(expected) == len(observed)
    # can't do this owing to randomness of results
    # for e, o in zip(expected, observed):
    #     asert e == o
    for term in observed:
        assert term == term.lower()


def test_textrank_norm_none(spacy_doc):
    expected = ["Friedman", "Beirut", "New", "Arab", "Award"]
    observed = [
        term for term, _ in keyterms.textrank(spacy_doc, normalize=None, n_keyterms=5)
    ]
    assert len(expected) == len(observed)
    # can't do this owing to randomness of results
    # for e, o in zip(expected, observed):
    #     asert e == o


def test_textrank_norm_normalized_str(spacy_doc):
    expected = ["Friedman", "Beirut", "New", "Award", "foreign"]
    observed = [
        term
        for term, _ in keyterms.textrank(
            spacy_doc, normalize=spacy_utils.get_normalized_text, n_keyterms=5
        )
    ]
    assert len(expected) == len(observed)
    # can't do this owing to randomness of results
    # for e, o in zip(expected, observed):
    #     asert e == o


def test_singlegrank(spacy_doc):
    expected = [
        "new york times jerusalem bureau",
        "new york times",
        "friedman",
        "foreign reporting",
        "international reporting",
        "pulitzer prize",
        "book award",
        "press international",
        "president george",
        "beirut",
    ]
    observed = [term for term, _ in keyterms.singlerank(spacy_doc)]
    assert len(expected) == len(observed)
    # can't do this owing to randomness of results
    # for e, o in zip(expected, observed):
    #     asert e == o


def test_singlegrank_n_keyterms(spacy_doc):
    expected = [
        "new york times jerusalem bureau",
        "new york times",
        "friedman",
        "foreign reporting",
        "international reporting",
    ]
    observed = [term for term, _ in keyterms.singlerank(spacy_doc, n_keyterms=5)]
    assert len(expected) == len(observed)
    # can't do this owing to randomness of results
    # for e, o in zip(expected, observed):
    #     asert e == o


def test_singlegrank_norm_lower(spacy_doc):
    expected = [
        "new york times jerusalem bureau",
        "new york times",
        "friedman",
        "foreign reporting",
        "international reporting",
    ]
    observed = [
        term
        for term, _ in keyterms.singlerank(spacy_doc, normalize="lower", n_keyterms=5)
    ]
    assert len(expected) == len(observed)
    # can't do this owing to randomness of results
    # for e, o in zip(expected, observed):
    #     asert e == o
    for term in observed:
        assert term == term.lower()


def test_singlegrank_norm_none(spacy_doc):
    expected = [
        "New York Times Jerusalem",
        "New York Times",
        "Friedman",
        "Pulitzer Prize",
        "foreign reporting",
    ]
    observed = [
        term for term, _ in keyterms.singlerank(spacy_doc, normalize=None, n_keyterms=5)
    ]
    assert len(expected) == len(observed)
    # can't do this owing to randomness of results
    # for e, o in zip(expected, observed):
    #     asert e == o


def test_singlegrank_norm_normalized_str(spacy_doc):
    expected = [
        "New York Times Jerusalem",
        "New York Times",
        "Friedman",
        "Pulitzer Prize",
        "foreign reporting",
    ]
    observed = [
        term
        for term, _ in keyterms.singlerank(
            spacy_doc, normalize=spacy_utils.get_normalized_text, n_keyterms=5
        )
    ]
    assert len(expected) == len(observed)
    # can't do this owing to randomness of results
    # for e, o in zip(expected, observed):
    #     asert e == o


def test_key_terms_from_semantic_network(spacy_doc):
    # let's just make sure that these run without exception
    _ = keyterms.key_terms_from_semantic_network(spacy_doc, ranking_algo="divrank")
    _ = keyterms.key_terms_from_semantic_network(spacy_doc, ranking_algo="bestcoverage")


def test_key_terms_from_semantic_network_empty(empty_spacy_doc):
    key_terms = keyterms.key_terms_from_semantic_network(empty_spacy_doc)
    assert isinstance(key_terms, list) is True
    assert len(key_terms) == 0


def test_most_discriminating_terms(spacy_doc):
    text1 = """Friedman joined the London bureau of United Press International after completing his master's degree. He was dispatched a year later to Beirut, where he lived from June 1979 to May 1981 while covering the Lebanon Civil War. He was hired by The New York Times as a reporter in 1981 and re-dispatched to Beirut at the start of the 1982 Israeli invasion of Lebanon. His coverage of the war, particularly the Sabra and Shatila massacre, won him the Pulitzer Prize for International Reporting (shared with Loren Jenkins of The Washington Post). Alongside David K. Shipler he also won the George Polk Award for foreign reporting.
    In June 1984, Friedman was transferred to Jerusalem, where he served as the New York Times Jerusalem Bureau Chief until February 1988. That year he received a second Pulitzer Prize for International Reporting, which cited his coverage of the First Palestinian Intifada. He wrote a book, From Beirut to Jerusalem, describing his experiences in the Middle East, which won the 1989 U.S. National Book Award for Nonfiction.
    Friedman covered Secretary of State James Baker during the administration of President George H. W. Bush. Following the election of Bill Clinton in 1992, Friedman became the White House correspondent for the New York Times. In 1994, he began to write more about foreign policy and economics, and moved to the op-ed page of The New York Times the following year as a foreign affairs columnist. In 2002, Friedman won the Pulitzer Prize for Commentary for his "clarity of vision, based on extensive reporting, in commenting on the worldwide impact of the terrorist threat."
    In February 2002, Friedman met Saudi Crown Prince Abdullah and encouraged him to make a comprehensive attempt to end the Arab-Israeli conflict by normalizing Arab relations with Israel in exchange for the return of refugees alongside an end to the Israel territorial occupations. Abdullah proposed the Arab Peace Initiative at the Beirut Summit that March, which Friedman has since strongly supported.
    Friedman received the 2004 Overseas Press Club Award for lifetime achievement and was named to the Order of the British Empire by Queen Elizabeth II.
    In May 2011, The New York Times reported that President Barack Obama "has sounded out" Friedman concerning Middle East issues."""

    text2 = """In 1954, Bucksbaum and his brother Martin borrowed $1.2 million and built the first shopping center in Cedar Rapids, Iowa, anchored by a fourth family grocery store.
    They expanded into enclosed malls which mirrored the continued movement to the suburbs seen in the 1960s.
    By 1964, their company - then named General Management - owned five malls anchored by the Younkers department store.
    In 1972, the company became publicly traded on the New York Stock Exchange under the name General Growth Properties and became the second-largest owner, developer, and manager of regional shopping malls in the country.
    Bucksbaum served as its Chairman and Chief Executive Officer and under his tenure, he formed two Real estate investment trusts and expanded the company's portfolio of malls and shopping centers via more than $36 billion in acquisitions.
    In 1984, General Growth sold 19 malls for $800 million to Equitable Real Estate, which was deemed the “nation’s largest single asset real estate transaction” to date.
    In 1995, his brother Martin died and he re-located the company to Chicago.
    In 2004, General Growth purchased The Rouse Company for $14.2 billion.
    By 2007, General Growth was the second-largest REIT owning 194 malls with over 200 million square feet in 44 states.
    In 2008, General Growth filed for Chapter 11 bankruptcy protection after the collapse of the stock market."""

    doc1 = [line.split() for line in text1.split("\n")]
    doc2 = [line.split() for line in text2.split("\n")]

    expected = (["Friedman", "Times"], ["General", "malls"])
    observed = keyterms.most_discriminating_terms(
        doc1 + doc2, [True] * len(doc1) + [False] * len(doc2), top_n_terms=2
    )

    print(observed)
    assert expected == observed
