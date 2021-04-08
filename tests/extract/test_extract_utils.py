import pytest

import textacy
from textacy.extract import utils


@pytest.fixture(scope="module")
def doc():
    lang = textacy.load_spacy_lang("en_core_web_sm")
    text = (
        "Many years later, as he faced the firing squad, Colonel Aureliano Buendía was "
        "to remember that distant afternoon when his father took him to discover ice. "
        "At that time Macondo was a village of twenty adobe houses, built on the bank "
        "of a river of clear water that ran along a bed of polished stones, which were "
        "white and enormous, like prehistoric eggs. The world was so recent that many "
        "things lacked names, and in order to indicate them it was necessary to point."
    )
    return textacy.make_spacy_doc(text, lang=lang)


@pytest.fixture(scope="module")
def term_tokens(doc):
    return list(textacy.extract.words(doc))


@pytest.fixture(scope="module")
def term_spans(doc):
    return list(textacy.extract.ngrams(doc, 2))


class TestTermsToStrings:

    def test_term_spans(self, term_spans):
        results = list(utils.terms_to_strings(term_spans, "orth"))
        assert results
        assert isinstance(results[0], str)

    def test_term_tokens(self, term_tokens):
        results = list(utils.terms_to_strings(term_tokens, "orth"))
        assert results
        assert isinstance(results[0], str)

    @pytest.mark.parametrize("by", ["orth", "lower", "lemma"])
    def test_by_str(self, by, term_spans):
        results = list(utils.terms_to_strings(term_spans, by))
        assert results
        assert isinstance(results[0], str)

    @pytest.mark.parametrize("by", [lambda term: term.text])
    def test_by_callable(self, by, term_spans):
        results = list(utils.terms_to_strings(term_spans, by))
        assert results
        assert isinstance(results[0], str)

    @pytest.mark.parametrize("by", ["LEMMA", None, True])
    def test_by_invalid(self, by, term_spans):
        with pytest.raises(ValueError):
            _ = list(utils.terms_to_strings(term_spans, by))


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
    result1 = utils.aggregate_term_variants(terms)
    result2 = utils.aggregate_term_variants(
        terms, acro_defs={"BJD": "Burton Jacob DeWilde"})
    assert len(result2) <= len(result1) <= len(terms)


# def test_most_discriminating_terms():
#     text1 = """Friedman joined the London bureau of United Press International after completing his master's degree. He was dispatched a year later to Beirut, where he lived from June 1979 to May 1981 while covering the Lebanon Civil War. He was hired by The New York Times as a reporter in 1981 and re-dispatched to Beirut at the start of the 1982 Israeli invasion of Lebanon. His coverage of the war, particularly the Sabra and Shatila massacre, won him the Pulitzer Prize for International Reporting (shared with Loren Jenkins of The Washington Post). Alongside David K. Shipler he also won the George Polk Award for foreign reporting.
#     In June 1984, Friedman was transferred to Jerusalem, where he served as the New York Times Jerusalem Bureau Chief until February 1988. That year he received a second Pulitzer Prize for International Reporting, which cited his coverage of the First Palestinian Intifada. He wrote a book, From Beirut to Jerusalem, describing his experiences in the Middle East, which won the 1989 U.S. National Book Award for Nonfiction.
#     Friedman covered Secretary of State James Baker during the administration of President George H. W. Bush. Following the election of Bill Clinton in 1992, Friedman became the White House correspondent for the New York Times. In 1994, he began to write more about foreign policy and economics, and moved to the op-ed page of The New York Times the following year as a foreign affairs columnist. In 2002, Friedman won the Pulitzer Prize for Commentary for his "clarity of vision, based on extensive reporting, in commenting on the worldwide impact of the terrorist threat."
#     In February 2002, Friedman met Saudi Crown Prince Abdullah and encouraged him to make a comprehensive attempt to end the Arab-Israeli conflict by normalizing Arab relations with Israel in exchange for the return of refugees alongside an end to the Israel territorial occupations. Abdullah proposed the Arab Peace Initiative at the Beirut Summit that March, which Friedman has since strongly supported.
#     Friedman received the 2004 Overseas Press Club Award for lifetime achievement and was named to the Order of the British Empire by Queen Elizabeth II.
#     In May 2011, The New York Times reported that President Barack Obama "has sounded out" Friedman concerning Middle East issues."""

#     text2 = """In 1954, Bucksbaum and his brother Martin borrowed $1.2 million and built the first shopping center in Cedar Rapids, Iowa, anchored by a fourth family grocery store.
#     They expanded into enclosed malls which mirrored the continued movement to the suburbs seen in the 1960s.
#     By 1964, their company - then named General Management - owned five malls anchored by the Younkers department store.
#     In 1972, the company became publicly traded on the New York Stock Exchange under the name General Growth Properties and became the second-largest owner, developer, and manager of regional shopping malls in the country.
#     Bucksbaum served as its Chairman and Chief Executive Officer and under his tenure, he formed two Real estate investment trusts and expanded the company's portfolio of malls and shopping centers via more than $36 billion in acquisitions.
#     In 1984, General Growth sold 19 malls for $800 million to Equitable Real Estate, which was deemed the “nation’s largest single asset real estate transaction” to date.
#     In 1995, his brother Martin died and he re-located the company to Chicago.
#     In 2004, General Growth purchased The Rouse Company for $14.2 billion.
#     By 2007, General Growth was the second-largest REIT owning 194 malls with over 200 million square feet in 44 states.
#     In 2008, General Growth filed for Chapter 11 bankruptcy protection after the collapse of the stock market."""

#     doc1 = [line.split() for line in text1.split("\n")]
#     doc2 = [line.split() for line in text2.split("\n")]

#     expected = (["Friedman", "Times"], ["General", "malls"])
#     observed = utils.most_discriminating_terms(
#         doc1 + doc2, [True] * len(doc1) + [False] * len(doc2), top_n_terms=2
#     )
#     assert expected == observed


@pytest.mark.parametrize(
    "input_,output_",
    [
        (
            ["foo (bar)", "foo?", "bar!", "-123.4"],
            ["foo (bar)", "foo?", "bar!", "-123.4"],
        ),
        (["(foo bar", "foo) bar", "?>,!-.", "", "foo) (bar"], []),
        (
            [
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
            ],
            [
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
            ],
        ),
    ],
)
def test_clean_terms(input_, output_):
    assert list(utils.clean_terms(input_)) == output_
