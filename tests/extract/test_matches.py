import re

import pytest
from spacy.tokens import Span

from textacy import extract
from textacy.extract.matches import _make_pattern_from_string


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
    def test_pattern_types(self, doc_en, patterns):
        matches = list(extract.token_matches(doc_en, patterns))[:5]
        assert matches
        assert all(isinstance(span, Span) for span in matches)

    def test_patstr(self, doc_en):
        matches = list(extract.token_matches(doc_en, "POS:NOUN"))[:5]
        assert matches
        assert all(len(span) == 1 for span in matches)
        assert all(span[0].pos_ == "NOUN" for span in matches)

    def test_patstr_op(self, doc_en):
        matches = list(extract.token_matches(doc_en, "POS:NOUN:+"))[:5]
        assert matches
        assert all(len(span) >= 1 for span in matches)
        assert all(tok.pos_ == "NOUN" for span in matches for tok in span)

    def test_patstr_bool(self, doc_en):
        matches = list(extract.token_matches(doc_en, "IS_PUNCT:bool(True)"))[:5]
        assert matches
        assert all(span[0].is_punct is True for span in matches)

    def test_patstr_int(self, doc_en):
        matches = list(extract.token_matches(doc_en, "LENGTH:int(5)"))[:5]
        assert matches
        assert all(len(span[0]) == 5 for span in matches)

    def test_patdict(self, doc_en):
        matches = list(extract.token_matches(doc_en, [{"POS": "NOUN"}]))[:5]
        assert matches
        assert all(len(span) == 1 for span in matches)
        assert all(span[0].pos_ == "NOUN" for span in matches)

    def test_patdict_op(self, doc_en):
        matches = list(extract.token_matches(doc_en, [{"POS": "NOUN", "OP": "+"}]))[:5]
        assert matches
        assert all(len(span) >= 1 for span in matches)
        assert all(tok.pos_ == "NOUN" for span in matches for tok in span)

    def test_patdict_bool(self, doc_en):
        matches = list(extract.token_matches(doc_en, [{"IS_PUNCT": True}]))[:5]
        assert matches
        assert all(span[0].is_punct is True for span in matches)

    def test_patdict_int(self, doc_en):
        matches = list(extract.token_matches(doc_en, [{"LENGTH": 5}]))[:5]
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
        ],
    )
    def test_pattern(self, pattern, doc_en):
        matches = list(extract.regex_matches(doc_en, pattern))
        assert matches
        assert all(isinstance(match, Span) for match in matches)

    @pytest.mark.parametrize("alignment_mode", ["strict", "contract", "expand"])
    def test_alignment_mode(self, alignment_mode, doc_en):
        pattern = r"\w [Tt]he \w"
        matches = list(
            extract.regex_matches(doc_en, pattern, alignment_mode=alignment_mode)
        )
        if alignment_mode == "strict":
            assert not matches
        elif alignment_mode == "contract":
            assert all(match.text.lower() == "the" for match in matches)
        elif alignment_mode == "expand":
            assert all("the" in match.text.lower() for match in matches)
            assert all(len(match.text) > 3 for match in matches)
