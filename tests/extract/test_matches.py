import re

import pytest
from spacy.tokens import Span

from textacy import load_spacy_lang
from textacy import extract
from textacy.extract.matches import _make_pattern_from_string


@pytest.fixture(scope="module")
def doc():
    nlp = load_spacy_lang("en_core_web_sm")
    text = (
        "Many years later, as he faced the firing squad, Colonel Aureliano Buendía was "
        "to remember that distant afternoon when his father took him to discover ice. "
        "At that time Macondo was a village of twenty adobe houses, built on the bank "
        "of a river of clear water that ran along a bed of polished stones, which were "
        "white and enormous, like prehistoric eggs. The world was so recent that many "
        "things lacked names, and in order to indicate them it was necessary to point."
    )
    return nlp(text)


class TestTokenMatches:

    @pytest.mark.parametrize(
        "patterns",
        [
            "POS:NOUN",
            ["POS:NOUN", "POS:DET"],
            [{"POS": "NOUN"}],
            [[{"POS": "NOUN"}], [{"POS": "DET"}]],
        ],
    )
    def test_pattern_types(self, doc, patterns):
        matches = list(extract.token_matches(doc, patterns))[:5]
        assert matches
        assert all(isinstance(span, Span) for span in matches)

    def test_patstr(self, doc):
        matches = list(extract.token_matches(doc, "POS:NOUN"))[:5]
        assert matches
        assert all(len(span) == 1 for span in matches)
        assert all(span[0].pos_ == "NOUN" for span in matches)

    def test_patstr_op(self, doc):
        matches = list(extract.token_matches(doc, "POS:NOUN:+"))[:5]
        assert matches
        assert all(len(span) >= 1 for span in matches)
        assert all(tok.pos_ == "NOUN" for span in matches for tok in span)

    def test_patstr_bool(self, doc):
        matches = list(extract.token_matches(doc, "IS_PUNCT:bool(True)"))[:5]
        assert matches
        assert all(span[0].is_punct is True for span in matches)

    def test_patstr_int(self, doc):
        matches = list(extract.token_matches(doc, "LENGTH:int(5)"))[:5]
        assert matches
        assert all(len(span[0]) == 5 for span in matches)

    def test_patdict(self, doc):
        matches = list(extract.token_matches(doc, [{"POS": "NOUN"}]))[:5]
        assert matches
        assert all(len(span) == 1 for span in matches)
        assert all(span[0].pos_ == "NOUN" for span in matches)

    def test_patdict_op(self, doc):
        matches = list(extract.token_matches(doc, [{"POS": "NOUN", "OP": "+"}]))[:5]
        assert matches
        assert all(len(span) >= 1 for span in matches)
        assert all(tok.pos_ == "NOUN" for span in matches for tok in span)

    def test_patdict_bool(self, doc):
        matches = list(extract.token_matches(doc, [{"IS_PUNCT": True}]))[:5]
        assert matches
        assert all(span[0].is_punct is True for span in matches)

    def test_patdict_int(self, doc):
        matches = list(extract.token_matches(doc, [{"LENGTH": 5}]))[:5]
        assert matches
        assert all(len(span[0]) == 5 for span in matches)

    @pytest.mark.parametrize(
        "patstr, pat",
        [
            ("TAG:VBZ", [{"TAG": "VBZ"}]),
            ("POS:NOUN:+", [{"POS": "NOUN", "OP": "+"}]),
            ("IS_PUNCT:bool(False)", [{"IS_PUNCT": False}]),
            (
                "IS_DIGIT:bool(True):? POS:NOUN:*",
                [{"IS_DIGIT": True, "OP": "?"}, {"POS": "NOUN", "OP": "*"}],
            ),
            (
                "LENGTH:int(5) DEP:nsubj:!",
                [{"LENGTH": 5}, {"DEP": "nsubj", "OP": "!"}],
            ),
            ("POS:DET :", [{"POS": "DET"}, {}]),
            (
                "IS_PUNCT:bool(False) : IS_PUNCT:bool(True)",
                [{"IS_PUNCT": False}, {}, {"IS_PUNCT": True}],
            ),
        ],
    )
    def test_make_pattern_from_string(self, patstr, pat):
        assert _make_pattern_from_string(patstr) == pat

    @pytest.mark.parametrize("patstr", ["POS", "POS:NOUN:VERB:+", "POS:NOUN:*?"])
    def test_make_pattern_from_str_error(self, patstr):
        with pytest.raises(ValueError):
            _ = _make_pattern_from_string(patstr)


class TestRegexMatches:

    @pytest.mark.parametrize(
        "pattern",
        [
            r"\w+ was",
            re.compile(r"\w+ was"),
            r"[Tt]he",
        ]
    )
    def test_pattern(self, pattern, doc):
        matches = list(extract.regex_matches(doc, pattern))
        assert matches
        assert all(isinstance(match, Span) for match in matches)

    @pytest.mark.parametrize("alignment_mode", ["strict", "contract", "expand"])
    def test_alignment_mode(self, alignment_mode, doc):
        pattern = r"\w [Tt]he \w"
        matches = list(
            extract.regex_matches(doc, pattern, alignment_mode=alignment_mode)
        )
        if alignment_mode == "strict":
            assert not matches
        elif alignment_mode == "contract":
            assert all(match.text.lower() == "the" for match in matches)
        elif alignment_mode == "expand":
            assert all("the" in match.text.lower() for match in matches)
            assert all(len(match.text) > 3 for match in matches)
