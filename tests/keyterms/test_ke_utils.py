# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

from textacy import ke


def test_aggregate_term_variants():
    # TODO: the actual results are NOT what i'd expect; figure out why
    terms = set([
        "vice versa",
        "vice-versa",
        "vice/versa",
        "BJD",
        "Burton Jacob DeWilde",
        "the big black cat named Rico",
        "the black cat named Rico",
    ])
    result1 = ke.utils.aggregate_term_variants(terms)
    result2 = ke.utils.aggregate_term_variants(
        terms, acro_defs={"BJD": "Burton Jacob DeWilde"})
    assert len(result2) <= len(result1) <= len(terms)


def test_most_discriminating_terms():
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
    observed = ke.utils.most_discriminating_terms(
        doc1 + doc2, [True] * len(doc1) + [False] * len(doc2), top_n_terms=2
    )
    assert expected == observed
