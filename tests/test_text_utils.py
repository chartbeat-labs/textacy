import pytest

from textacy import text_utils


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


@pytest.mark.parametrize(
    "token",
    [
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
)
def test_is_acronym_good(token):
    assert text_utils.is_acronym(token)


@pytest.mark.parametrize(
    "token",
    [
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
)
def test_is_acronym_bad(token):
    assert not text_utils.is_acronym(token)


@pytest.mark.parametrize(
    "token,exclude,expected",
    [
        ("NASA", {"NASA"}, False),
        ("NASA", {"CSA", "ISS"}, True),
        ("NASA", None, True)
    ]
)
def test_is_acronym_exclude(token, exclude, expected):
    assert text_utils.is_acronym(token, exclude=exclude) == expected


@pytest.mark.parametrize(
    "text,keyword,ignore_case,window_width,has_results",
    [
        (TEXT, "clinton", True, 50, True),
        (TEXT, "clinton", False, 50, False),
        (TEXT, "clinton", True, 10, True),
        (TEXT, "all", True, 50, True),
        (TEXT, "All", False, 50, True),
        (
            "No llores porque ya se terminó, sonríe porque sucedió.",
            "terminó",
            True,
            50,
            True,
        ),
    ],
)
def test_keyword_in_context(text, keyword, ignore_case, window_width, has_results):
    results = list(
        text_utils.keyword_in_context(
            text,
            keyword,
            ignore_case=ignore_case,
            window_width=window_width,
            print_only=False,
        )
    )
    # check if any results
    if has_results:
        assert results
    else:
        assert not results
    for pre, kw, post in results:
        # check kw match by case
        if ignore_case is True:
            assert kw.lower() == keyword.lower()
        else:
            assert kw == keyword
        # check pre/post window widths
        assert len(pre) <= window_width
        assert len(post) <= window_width


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
    assert list(text_utils.clean_terms(input_)) == output_
