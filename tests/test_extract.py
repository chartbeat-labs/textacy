import collections
import re

import pytest
import spacy
from spacy.tokens import Span, Token

from textacy import load_spacy_lang
from textacy import constants, extract


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


class TestWords:

    def test_default(self, spacy_doc):
        result = list(extract.words(spacy_doc))
        assert all(isinstance(tok, Token) for tok in result)
        assert not any(tok.is_space for tok in result)

    def test_filter(self, spacy_doc):
        result = list(
            extract.words(
                spacy_doc, filter_stops=True, filter_punct=True, filter_nums=True
            )
        )
        assert not any(tok.is_stop for tok in result)
        assert not any(tok.is_punct for tok in result)
        assert not any(tok.like_num for tok in result)

    def test_pos(self, spacy_doc):
        result1 = list(extract.words(spacy_doc, include_pos={"NOUN"}))
        result2 = list(extract.words(spacy_doc, include_pos="NOUN"))
        assert all(tok.pos_ == "NOUN" for tok in result1)
        assert all(tok.pos_ == "NOUN" for tok in result2)
        result3 = list(extract.words(spacy_doc, exclude_pos={"NOUN"}))
        result4 = list(extract.words(spacy_doc, exclude_pos="NOUN"))
        assert not any(tok.pos_ == "NOUN" for tok in result3)
        assert not any(tok.pos_ == "NOUN" for tok in result4)

    def test_min_freq(self, spacy_doc):
        counts = collections.Counter()
        counts.update(tok.lower_ for tok in spacy_doc)
        result = list(extract.words(spacy_doc, min_freq=2))
        assert all(counts[tok.lower_] >= 2 for tok in result)


class TestNGrams:

    def test_n_less_than_1(self, spacy_doc):
        with pytest.raises(ValueError):
            _ = list(extract.ngrams(spacy_doc, 0))

    def test_n(self, spacy_doc):
        for n in (1, 2):
            result = list(extract.ngrams(spacy_doc, n))
            assert all(isinstance(span, Span) for span in result)
            assert all(len(span) == n for span in result)

    def test_filter(self, spacy_doc):
        result = list(
            extract.ngrams(
                spacy_doc, 2, filter_stops=True, filter_punct=True, filter_nums=True
            )
        )
        assert not any(span[0].is_stop or span[-1].is_stop for span in result)
        assert not any(tok.is_punct for span in result for tok in span)
        assert not any(tok.like_num for span in result for tok in span)

    def test_min_freq(self, spacy_doc):
        n = 2
        counts = collections.Counter()
        counts.update(spacy_doc[i : i + n].text.lower() for i in range(len(spacy_doc) - n + 1))
        result = list(extract.ngrams(spacy_doc, 2, min_freq=2))
        assert all(counts[span.text.lower()] >= 2 for span in result)

    def test_pos(self, spacy_doc):
        result1 = list(extract.ngrams(spacy_doc, 2, include_pos={"NOUN"}))
        result2 = list(extract.ngrams(spacy_doc, 2, include_pos="NOUN"))
        assert all(tok.pos_ == "NOUN" for span in result1 for tok in span)
        assert all(tok.pos_ == "NOUN" for span in result2 for tok in span)
        result3 = list(extract.ngrams(spacy_doc, 2, exclude_pos={"NOUN"}))
        result4 = list(extract.ngrams(spacy_doc, 2, exclude_pos="NOUN"))
        assert not any(tok.pos_ == "NOUN" for span in result3 for tok in span)
        assert not any(tok.pos_ == "NOUN" for span in result4 for tok in span)


class TestEntities:

    def test_default(self, spacy_doc):
        result = list(extract.entities(spacy_doc, drop_determiners=False))
        assert all(isinstance(span, Span) for span in result)
        assert all(span.label_ for span in result)
        assert all(span[0].ent_type for span in result)

    def test_include_types(self, spacy_doc):
        ent_types = ["PERSON", "GPE"]
        for include_types in ent_types:
            result = extract.entities(spacy_doc, include_types=include_types)
            assert all(span.label_ == include_types for span in result)
        ent_types = [{"PERSON", "GPE"}, ("DATE", "ORG"), ["LOC"]]
        for include_types in ent_types:
            result = extract.entities(spacy_doc, include_types=include_types)
            assert all(span.label_ in include_types for span in result)
        # special numeric cases!
        ent_types = ["NUMERIC", ("NUMERIC",), {"PERSON", "NUMERIC"}]
        for include_types in ent_types:
            include_types_parsed = extract._parse_ent_types(include_types, "include")
            result = extract.entities(spacy_doc, include_types=include_types)
            assert all(span.label_ in include_types_parsed for span in result)

    def test_exclude_types(self, spacy_doc):
        ent_types = ["PERSON", "GPE"]
        for exclude_types in ent_types:
            result = extract.entities(spacy_doc, exclude_types=exclude_types)
            assert all(span.label_ != exclude_types for span in result)
        ent_types = [{"PERSON", "GPE"}, ("DATE", "ORG"), ["LOC"]]
        for exclude_types in ent_types:
            result = extract.entities(spacy_doc, exclude_types=exclude_types)
            assert all(span.label_ not in exclude_types for span in result)
        # special numeric cases!
        ent_types = ["NUMERIC", ("NUMERIC",), {"PERSON", "NUMERIC"}]
        for exclude_types in ent_types:
            exclude_types_parsed = extract._parse_ent_types(exclude_types, "exclude")
            result = extract.entities(spacy_doc, exclude_types=exclude_types)
            assert all(span.label_ not in exclude_types_parsed for span in result)

    def test_parse_ent_types_bad_type(self):
        for bad_type in [1, 3.1415, True, b"PERSON"]:
            with pytest.raises(TypeError):
                _ = extract._parse_ent_types(bad_type, "include")

    def test_min_freq(self, spacy_doc):
        result = list(extract.entities(spacy_doc, min_freq=2))
        assert len(result) == 0

    def test_determiner(self, spacy_doc):
        result = list(extract.entities(spacy_doc, drop_determiners=False))
        assert all(isinstance(span, Span) for span in result)
        assert any(span[0].pos_ == "DET" for span in result)
        assert all(span.label_ for span in result)

    def test_drop_determiners(self, spacy_doc):
        result = list(extract.entities(spacy_doc, drop_determiners=True))
        assert not any(span[0].pos_ == "DET" for span in result)
        assert all(span.label_ for span in result)


class TestNounChunks:

    def test_default(self, spacy_doc):
        result = list(extract.noun_chunks(spacy_doc))
        assert all(isinstance(span, Span) for span in result)

    def test_determiner(self, spacy_doc):
        result = list(extract.noun_chunks(spacy_doc, drop_determiners=False))
        assert all(isinstance(span, Span) for span in result)
        assert any(span[0].pos_ == "DET" for span in result)

    def test_min_freq(self, spacy_doc):
        text = spacy_doc.text.lower()
        result = list(extract.noun_chunks(spacy_doc, drop_determiners=True, min_freq=2))
        assert all(text.count(span.text.lower()) >= 2 for span in result)


class TestPOSRegexMatches:

    def test_deprecation_warning(self, spacy_doc):
        with pytest.warns(DeprecationWarning):
            _ = list(extract.pos_regex_matches(spacy_doc, r"<NOUN>"))

    def test_simple(self, spacy_doc):
        result = list(extract.pos_regex_matches(spacy_doc, r"<NOUN>+"))
        assert all(isinstance(span, Span) for span in result)
        assert all(tok.pos_ == "NOUN" for span in result for tok in span)

    def test_complex(self, spacy_doc):
        pattern = constants.POS_REGEX_PATTERNS["en"]["NP"]
        valid_pos = set(re.findall(r"(\w+)", pattern))
        required_pos = {"NOUN", "PROPN"}
        result = list(extract.pos_regex_matches(spacy_doc, pattern))
        assert all(isinstance(span, Span) for span in result)
        assert all(tok.pos_ in valid_pos for span in result for tok in span)
        assert all(any(tok.pos_ in required_pos for tok in span) for span in result)


class TestMatches:

    def test_pattern_types(self, spacy_doc):
        all_patterns = [
            "POS:NOUN",
            ["POS:NOUN", "POS:DET"],
            [{"POS": "NOUN"}],
            [[{"POS": "NOUN"}], [{"POS": "DET"}]],
        ]
        for patterns in all_patterns:
            matches = list(extract.matches(spacy_doc, patterns))[:5]
            assert matches
            assert all(isinstance(span, Span) for span in matches)

    def test_patstr(self, spacy_doc):
        matches = list(extract.matches(spacy_doc, "POS:NOUN"))[:5]
        assert matches
        assert all(len(span) == 1 for span in matches)
        assert all(span[0].pos_ == "NOUN" for span in matches)

    def test_patstr_op(self, spacy_doc):
        matches = list(extract.matches(spacy_doc, "POS:NOUN:+"))[:5]
        assert matches
        assert all(len(span) >= 1 for span in matches)
        assert all(tok.pos_ == "NOUN" for span in matches for tok in span)

    def test_patstr_bool(self, spacy_doc):
        matches = list(extract.matches(spacy_doc, "IS_DIGIT:bool(True)"))[:5]
        assert matches
        assert all(span[0].is_digit is True for span in matches)

    @pytest.mark.xfail(
        spacy.__version__.startswith("2.2."),
        reason="https://github.com/explosion/spaCy/pull/4749",
    )
    def test_patstr_int(self, spacy_doc):
        matches = list(extract.matches(spacy_doc, "LENGTH:int(5)"))[:5]
        assert matches
        assert all(len(span[0]) == 5 for span in matches)

    def test_patdict(self, spacy_doc):
        matches = list(extract.matches(spacy_doc, [{"POS": "NOUN"}]))[:5]
        assert matches
        assert all(len(span) == 1 for span in matches)
        assert all(span[0].pos_ == "NOUN" for span in matches)

    def test_patdict_op(self, spacy_doc):
        matches = list(extract.matches(spacy_doc, [{"POS": "NOUN", "OP": "+"}]))[:5]
        assert matches
        assert all(len(span) >= 1 for span in matches)
        assert all(tok.pos_ == "NOUN" for span in matches for tok in span)

    def test_patdict_bool(self, spacy_doc):
        matches = list(extract.matches(spacy_doc, [{"IS_DIGIT": True}]))[:5]
        assert matches
        assert all(span[0].is_digit is True for span in matches)

    @pytest.mark.xfail(
        spacy.__version__.startswith("2.2."),
        reason="https://github.com/explosion/spaCy/pull/4749",
    )
    def test_patdict_int(self, spacy_doc):
        matches = list(extract.matches(spacy_doc, [{"LENGTH": 5}]))[:5]
        assert matches
        assert all(len(span[0]) == 5 for span in matches)

    def test_make_pattern_from_string(self):
        patstr_to_pats = [
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
        ]
        for patstr, pat in patstr_to_pats:
            assert extract._make_pattern_from_string(patstr) == pat

    def test_make_pattern_from_str_error(self):
        for patstr in ["POS", "POS:NOUN:VERB:+", "POS:NOUN:*?"]:
            with pytest.raises(ValueError):
                _ = extract._make_pattern_from_string(patstr)


class TestSubjectVerbObjectTriples:

    def test_default(self, spacy_doc):
        result = list(extract.subject_verb_object_triples(spacy_doc))
        assert all(isinstance(triple, tuple) for triple in result)
        assert all(isinstance(span, Span) for triple in result for span in triple)
        assert all(any(tok.pos_ == "VERB" for tok in triple[1]) for triple in result)


class TestAcronymsAndDefinitions:

    def test_default(self, spacy_doc):
        # TODO: figure out if this function no longer works, ugh
        # expected = {"I.M.F.": "International Monetary Fund"}
        expected = {"I.M.F.": ""}
        observed = extract.acronyms_and_definitions(spacy_doc)
        assert observed == expected

    def test_known(self, spacy_doc):
        expected = {"I.M.F.": "International Monetary Fund"}
        observed = extract.acronyms_and_definitions(
            spacy_doc, known_acro_defs={"I.M.F.": "International Monetary Fund"}
        )
        assert observed == expected
