import collections

import pytest
from spacy.tokens import Span, Token

from textacy import extract
from textacy.extract import basics


class TestWords:
    def test_default(self, doc_en):
        result = list(extract.words(doc_en))
        assert all(isinstance(tok, Token) for tok in result)
        assert not any(tok.is_space for tok in result)

    def test_filter(self, doc_en):
        result = list(
            extract.words(doc_en, filter_stops=True, filter_punct=True, filter_nums=True)
        )
        assert not any(tok.is_stop for tok in result)
        assert not any(tok.is_punct for tok in result)
        assert not any(tok.like_num for tok in result)

    def test_pos(self, doc_en):
        result1 = list(extract.words(doc_en, include_pos={"NOUN"}))
        result2 = list(extract.words(doc_en, include_pos="NOUN"))
        assert all(tok.pos_ == "NOUN" for tok in result1)
        assert all(tok.pos_ == "NOUN" for tok in result2)
        result3 = list(extract.words(doc_en, exclude_pos={"NOUN"}))
        result4 = list(extract.words(doc_en, exclude_pos="NOUN"))
        assert not any(tok.pos_ == "NOUN" for tok in result3)
        assert not any(tok.pos_ == "NOUN" for tok in result4)

    def test_min_freq(self, doc_en):
        counts = collections.Counter()
        counts.update(tok.lower_ for tok in doc_en)
        result = list(extract.words(doc_en, min_freq=2))
        assert all(counts[tok.lower_] >= 2 for tok in result)


class TestNGrams:
    @pytest.mark.parametrize("n", [1, 2])
    def test_n(self, n, doc_en):
        result = list(extract.ngrams(doc_en, n))
        assert all(isinstance(span, Span) for span in result)
        assert all(len(span) == n for span in result)

    @pytest.mark.parametrize("ns", [[1, 2], [1, 2, 3]])
    def test_multiple_ns(self, ns, doc_en):
        result = list(extract.ngrams(doc_en, ns))
        assert all(isinstance(span, Span) for span in result)
        minn = min(ns)
        maxn = max(ns)
        assert all(minn <= len(span) <= maxn for span in result)

    def test_n_less_than_1(self, doc_en):
        with pytest.raises(ValueError):
            _ = list(extract.ngrams(doc_en, 0))

    def test_filter(self, doc_en):
        result = list(
            extract.ngrams(
                doc_en, 2, filter_stops=True, filter_punct=True, filter_nums=True
            )
        )
        assert not any(span[0].is_stop or span[-1].is_stop for span in result)
        assert not any(tok.is_punct for span in result for tok in span)
        assert not any(tok.like_num for span in result for tok in span)

    def test_min_freq(self, doc_en):
        n = 2
        counts = collections.Counter()
        counts.update(doc_en[i : i + n].text.lower() for i in range(len(doc_en) - n + 1))
        result = list(extract.ngrams(doc_en, 2, min_freq=2))
        assert all(counts[span.text.lower()] >= 2 for span in result)

    def test_pos(self, doc_en):
        result1 = list(extract.ngrams(doc_en, 2, include_pos={"NOUN"}))
        result2 = list(extract.ngrams(doc_en, 2, include_pos="NOUN"))
        assert all(tok.pos_ == "NOUN" for span in result1 for tok in span)
        assert all(tok.pos_ == "NOUN" for span in result2 for tok in span)
        result3 = list(extract.ngrams(doc_en, 2, exclude_pos={"NOUN"}))
        result4 = list(extract.ngrams(doc_en, 2, exclude_pos="NOUN"))
        assert not any(tok.pos_ == "NOUN" for span in result3 for tok in span)
        assert not any(tok.pos_ == "NOUN" for span in result4 for tok in span)


class TestEntities:
    def test_default(self, doc_en):
        result = list(extract.entities(doc_en, drop_determiners=False))
        assert all(isinstance(span, Span) for span in result)
        assert all(span.label_ for span in result)
        assert all(span[0].ent_type for span in result)

    def test_include_types(self, doc_en):
        ent_types = ["PERSON", "GPE"]
        for include_types in ent_types:
            result = extract.entities(doc_en, include_types=include_types)
            assert all(span.label_ == include_types for span in result)
        ent_types = [{"PERSON", "GPE"}, ("DATE", "ORG"), ["LOC"]]
        for include_types in ent_types:
            result = extract.entities(doc_en, include_types=include_types)
            assert all(span.label_ in include_types for span in result)
        # special numeric cases!
        ent_types = ["NUMERIC", ("NUMERIC",), {"PERSON", "NUMERIC"}]
        for include_types in ent_types:
            include_types_parsed = basics._parse_ent_types(include_types, "include")
            result = extract.entities(doc_en, include_types=include_types)
            assert all(span.label_ in include_types_parsed for span in result)

    def test_exclude_types(self, doc_en):
        ent_types = ["PERSON", "GPE"]
        for exclude_types in ent_types:
            result = extract.entities(doc_en, exclude_types=exclude_types)
            assert all(span.label_ != exclude_types for span in result)
        ent_types = [{"PERSON", "GPE"}, ("DATE", "ORG"), ["LOC"]]
        for exclude_types in ent_types:
            result = extract.entities(doc_en, exclude_types=exclude_types)
            assert all(span.label_ not in exclude_types for span in result)
        # special numeric cases!
        ent_types = ["NUMERIC", ("NUMERIC",), {"PERSON", "NUMERIC"}]
        for exclude_types in ent_types:
            exclude_types_parsed = basics._parse_ent_types(exclude_types, "exclude")
            result = extract.entities(doc_en, exclude_types=exclude_types)
            assert all(span.label_ not in exclude_types_parsed for span in result)

    def test_parse_ent_types_bad_type(self):
        for bad_type in [1, 3.1415, True, b"PERSON"]:
            with pytest.raises(TypeError):
                _ = basics._parse_ent_types(bad_type, "include")

    def test_min_freq(self, doc_en):
        result = list(extract.entities(doc_en, min_freq=2))
        assert len(result) == 0

    def test_determiner(self, doc_long_en):
        result = list(extract.entities(doc_long_en, drop_determiners=False))
        assert all(isinstance(span, Span) for span in result)
        assert any(span[0].pos_ == "DET" for span in result)
        assert all(span.label_ for span in result)

    def test_drop_determiners(self, doc_long_en):
        result = list(extract.entities(doc_long_en, drop_determiners=True))
        assert not any(span[0].pos_ == "DET" for span in result)
        assert all(span.label_ for span in result)


class TestNounChunks:
    def test_default(self, doc_en):
        result = list(extract.noun_chunks(doc_en))
        assert all(isinstance(span, Span) for span in result)

    def test_determiner(self, doc_en):
        result = list(extract.noun_chunks(doc_en, drop_determiners=False))
        assert all(isinstance(span, Span) for span in result)
        assert any(span[0].pos_ == "DET" for span in result)

    def test_min_freq(self, doc_en):
        text = doc_en.text.lower()
        result = list(extract.noun_chunks(doc_en, drop_determiners=True, min_freq=2))
        assert all(text.count(span.text.lower()) >= 2 for span in result)


class TestTerms:
    def test_default(self, doc_en):
        with pytest.raises(ValueError):
            _ = list(extract.terms(doc_en))

    def test_simple_args(self, doc_en):
        results = list(extract.terms(doc_en, ngs=2, ents=True, ncs=True))
        assert results
        assert all(isinstance(result, Span) for result in results)

    def test_callable_args(self, doc_en):
        results = list(
            extract.terms(
                doc_en,
                ngs=lambda doc: extract.ngrams(doc, n=2),
                ents=extract.entities,
                ncs=extract.noun_chunks,
            )
        )
        assert results
        assert all(isinstance(result, Span) for result in results)

    @pytest.mark.parametrize("dedupe", [True, False])
    def test_dedupe(self, dedupe, doc_en):
        results = list(extract.terms(doc_en, ngs=2, ents=True, ncs=True, dedupe=dedupe))
        assert results
        if dedupe is True:
            assert len(results) == len(
                set((result.start, result.end) for result in results)
            )
        else:
            assert len(results) > len(
                set((result.start, result.end) for result in results)
            )
