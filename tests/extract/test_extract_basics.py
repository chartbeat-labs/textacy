import collections

import pytest
from spacy.tokens import Span, Token

from textacy import load_spacy_lang
from textacy import extract
from textacy.extract import basics


@pytest.fixture(scope="module")
def spacy_doc(lang_en):
    text = (
        "Two weeks ago, I was in Kuwait participating in an I.M.F. (International Monetary Fund) seminar for Arab educators. "
        "For 30 minutes, we discussed the impact of technology trends on education in the Middle East. "
        "And then an Egyptian education official raised his hand and asked if he could ask me a personal question: \"I heard Donald Trump say we need to close mosques in the United States,\" he said with great sorrow. "
        "\"Is that what we want our kids to learn?\""
    )
    spacy_doc = lang_en(text)
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

    @pytest.mark.parametrize("n", [1, 2])
    def test_n(self, n, spacy_doc):
        result = list(extract.ngrams(spacy_doc, n))
        assert all(isinstance(span, Span) for span in result)
        assert all(len(span) == n for span in result)

    @pytest.mark.parametrize("ns", [[1, 2], [1, 2, 3]])
    def test_multiple_ns(self, ns, spacy_doc):
        result = list(extract.ngrams(spacy_doc, ns))
        assert all(isinstance(span, Span) for span in result)
        minn = min(ns)
        maxn = max(ns)
        assert all(minn <= len(span) <= maxn for span in result)

    def test_n_less_than_1(self, spacy_doc):
        with pytest.raises(ValueError):
            _ = list(extract.ngrams(spacy_doc, 0))

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
            include_types_parsed = basics._parse_ent_types(include_types, "include")
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
            exclude_types_parsed = basics._parse_ent_types(exclude_types, "exclude")
            result = extract.entities(spacy_doc, exclude_types=exclude_types)
            assert all(span.label_ not in exclude_types_parsed for span in result)

    def test_parse_ent_types_bad_type(self):
        for bad_type in [1, 3.1415, True, b"PERSON"]:
            with pytest.raises(TypeError):
                _ = basics._parse_ent_types(bad_type, "include")

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


class TestTerms:

    def test_default(self, spacy_doc):
        with pytest.raises(ValueError):
            _ = list(extract.terms(spacy_doc))

    def test_simple_args(self, spacy_doc):
        results = list(extract.terms(spacy_doc, ngs=2, ents=True, ncs=True))
        assert results
        assert all(isinstance(result, Span) for result in results)

    def test_callable_args(self, spacy_doc):
        results = list(
            extract.terms(
                spacy_doc,
                ngs=lambda doc: extract.ngrams(doc, n=2),
                ents=extract.entities,
                ncs=extract.noun_chunks,
            )
        )
        assert results
        assert all(isinstance(result, Span) for result in results)

    @pytest.mark.parametrize("dedupe", [True, False])
    def test_dedupe(self, dedupe, spacy_doc):
        results = list(extract.terms(spacy_doc, ngs=2, ents=True, ncs=True, dedupe=dedupe))
        assert results
        if dedupe is True:
            assert (
                len(results) ==
                len(set((result.start, result.end) for result in results))
            )
        else:
            assert (
                len(results) >
                len(set((result.start, result.end) for result in results))
            )
