from textacy.constants import RE_ACRONYM


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
BAD_ACRONYMS = ["A", "GHz", "1a", "D o E", "Ms", "Ph.D", "3-Dim."]


def test_good_acronym_regex():
    for item in GOOD_ACRONYMS:
        assert item == RE_ACRONYM.search(item).group()


def test_bad_acronym_regex():
    for item in BAD_ACRONYMS:
        assert RE_ACRONYM.search(item) is None
