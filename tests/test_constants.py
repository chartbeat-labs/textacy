import pytest

from textacy.constants import RE_ACRONYM


@pytest.mark.parametrize(
    "item",
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
def test_good_acronym_regex(item):
    assert item == RE_ACRONYM.search(item).group()


@pytest.mark.parametrize(
    "item",
    ["A", "GHz", "1a", "D o E", "Ms", "Ph.D", "3-Dim."]
)
def test_bad_acronym_regex(item):
    assert RE_ACRONYM.search(item) is None
