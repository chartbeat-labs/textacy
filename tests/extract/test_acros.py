import pytest
from spacy.tokens import Token

from textacy import extract


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
    ],
)
def test_is_acronym_good(token):
    assert extract.acros.is_acronym(token)


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
    ],
)
def test_is_acronym_bad(token):
    assert not extract.acros.is_acronym(token)


@pytest.mark.parametrize(
    "token,exclude,expected",
    [("NASA", {"NASA"}, False), ("NASA", {"CSA", "ISS"}, True), ("NASA", None, True)],
)
def test_is_acronym_exclude(token, exclude, expected):
    assert extract.acros.is_acronym(token, exclude=exclude) == expected


@pytest.mark.parametrize(
    "text, exp",
    [
        ("I want to work for NASA when I grow up, but not NOAA.", ["NASA", "NOAA"]),
        ("I want to live in the U.S. Do you?", ["U.S."]),
    ],
)
def test_acronyms(lang_en, text, exp):
    doc = lang_en(text)
    obs = list(extract.acronyms(doc))
    assert all(isinstance(tok, Token) for tok in obs)
    assert [tok.text for tok in obs] == exp


class TestAcronymsAndDefinitions:
    @pytest.mark.parametrize(
        "text, exp",
        [
            (
                "I want to work for NASA (National Aeronautics and Space Administration) when I grow up.",
                {"NASA": "National Aeronautics and Space Administration"},
            ),
            (
                "I want to work for NASA — National Aeronautics and Space Administration — when I grow up.",
                {"NASA": "National Aeronautics and Space Administration"},
            ),
            (
                "I want to work for NASA -- National Aeronautics and Space Administration -- when I grow up.",
                {"NASA": "National Aeronautics and Space Administration"},
            ),
            (
                "I want to work for NASA when I grow up.",
                {"NASA": ""},
            ),
            (
                "I want to live in the U.S.A. (United States of America) when I grow up.",
                # TODO: fix whatever error is failing to find the definition here
                # {"U.S.A.": "United States of America"},
                {"U.S.A.": ""},
            ),
            (
                "I want to be a W.A.S.P. (White Anglo Saxon Protestant) when I grow up.",
                # TODO: fix whatever error is failing to find the definition here
                # {"W.A.S.P": "White Anglo Saxon Protestant"},
                {"W.A.S.P.": ""},
            ),
        ],
    )
    def test_default(self, lang_en, text, exp):
        obs = extract.acronyms_and_definitions(lang_en(text))
        assert obs == exp

    @pytest.mark.parametrize(
        "text, known, exp",
        [
            (
                "I want to work for NASA when I grow up.",
                {"NASA": "National Aeronautics and Space Administration"},
                {"NASA": "National Aeronautics and Space Administration"},
            ),
            (
                "I want to work for NASA when I grow up.",
                {"NASA": "National Auto Sport Association"},
                {"NASA": "National Auto Sport Association"},
            ),
            (
                "I want to work for NASA (National Aeronautics and Space Administration) when I grow up.",
                {"NASA": "National Auto Sport Association"},
                {"NASA": "National Auto Sport Association"},
            ),
        ],
    )
    def test_known(self, lang_en, text, known, exp):
        obs = extract.acronyms_and_definitions(lang_en(text), known_acro_defs=known)
        assert obs == exp
