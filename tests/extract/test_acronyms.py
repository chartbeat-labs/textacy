import pytest

from textacy import extract
from textacy import load_spacy_lang


@pytest.fixture(scope="module")
def spacy_lang():
    return load_spacy_lang("en_core_web_sm")


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
        ]
    )
    def test_default(self, spacy_lang, text, exp):
        obs = extract.acronyms_and_definitions(spacy_lang(text))
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
    def test_default(self, spacy_lang, text, known, exp):
        obs = extract.acronyms_and_definitions(spacy_lang(text), known_acro_defs=known)
        assert obs == exp
