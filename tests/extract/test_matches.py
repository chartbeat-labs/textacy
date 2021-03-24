import pytest
from spacy.tokens import Span

from textacy import load_spacy_lang
from textacy import extract_
from textacy.extract_.matches import _make_pattern_from_string


@pytest.fixture(scope="module")
def spacy_doc():
    spacy_lang = load_spacy_lang("en_core_web_sm")
    text = (
        "Two weeks ago, I was in Kuwait participating in an I.M.F. (International Monetary Fund) seminar for Arab educators. "
        "For 30 minutes, we discussed the impact of technology trends on education in the Middle East. "
        "And then an Egyptian education official raised his hand and asked if he could ask me a personal question: \"I heard Donald Trump say we need to close mosques in the United States,\" he said with great sorrow. "
        "\"Is that what we want our kids to learn?\""
    )
    spacy_doc = spacy_lang(text)
    return spacy_doc


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
    def test_pattern_types(self, spacy_doc, patterns):
        matches = list(extract_.token_matches(spacy_doc, patterns))[:5]
        assert matches
        assert all(isinstance(span, Span) for span in matches)

    def test_patstr(self, spacy_doc):
        matches = list(extract_.token_matches(spacy_doc, "POS:NOUN"))[:5]
        assert matches
        assert all(len(span) == 1 for span in matches)
        assert all(span[0].pos_ == "NOUN" for span in matches)

    def test_patstr_op(self, spacy_doc):
        matches = list(extract_.token_matches(spacy_doc, "POS:NOUN:+"))[:5]
        assert matches
        assert all(len(span) >= 1 for span in matches)
        assert all(tok.pos_ == "NOUN" for span in matches for tok in span)

    def test_patstr_bool(self, spacy_doc):
        matches = list(extract_.token_matches(spacy_doc, "IS_DIGIT:bool(True)"))[:5]
        assert matches
        assert all(span[0].is_digit is True for span in matches)

    def test_patstr_int(self, spacy_doc):
        matches = list(extract_.token_matches(spacy_doc, "LENGTH:int(5)"))[:5]
        assert matches
        assert all(len(span[0]) == 5 for span in matches)

    def test_patdict(self, spacy_doc):
        matches = list(extract_.token_matches(spacy_doc, [{"POS": "NOUN"}]))[:5]
        assert matches
        assert all(len(span) == 1 for span in matches)
        assert all(span[0].pos_ == "NOUN" for span in matches)

    def test_patdict_op(self, spacy_doc):
        matches = list(extract_.token_matches(spacy_doc, [{"POS": "NOUN", "OP": "+"}]))[:5]
        assert matches
        assert all(len(span) >= 1 for span in matches)
        assert all(tok.pos_ == "NOUN" for span in matches for tok in span)

    def test_patdict_bool(self, spacy_doc):
        matches = list(extract_.token_matches(spacy_doc, [{"IS_DIGIT": True}]))[:5]
        assert matches
        assert all(span[0].is_digit is True for span in matches)

    def test_patdict_int(self, spacy_doc):
        matches = list(extract_.token_matches(spacy_doc, [{"LENGTH": 5}]))[:5]
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
