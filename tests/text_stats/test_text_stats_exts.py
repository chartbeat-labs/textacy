import pytest

import textacy
from textacy import text_stats


@pytest.mark.parametrize(
    "name", ["text_stats", "text_stats.basics", "text_stats.diversity"]
)
def test_set_get_remove_extensions(doc_en, name):
    textacy.set_doc_extensions(name)
    assert all(doc_en.has_extension(n) for n in textacy.get_doc_extensions(name).keys())
    if "." in name:
        assert not all(
            doc_en.has_extension(n) for n in textacy.get_doc_extensions("extract").keys()
        )
    textacy.remove_doc_extensions(name)
    assert not any(
        doc_en.has_extension(n) for n in textacy.get_doc_extensions(name).keys()
    )


@pytest.mark.parametrize(
    "ext_name, ext_type, kwargs",
    [
        ("n_sents", "property", None),
        ("n_words", "property", None),
        ("n_chars", "property", None),
        ("morph_counts", "property", None),
        ("tag_counts", "property", None),
        ("ttr", "method", {}),
        ("ttr", "method", {"variant": "root"}),
        ("flesch_reading_ease", "method", {}),
        ("flesch_reading_ease", "method", {"lang": "en"}),
    ],
)
def test_extensions_match(doc_en, ext_name, ext_type, kwargs):
    textacy.set_doc_extensions("text_stats")
    ext = getattr(doc_en._, ext_name)
    if ext_type == "property":
        func = textacy.get_doc_extensions("text_stats")[ext_name]["getter"]
        ext_val = ext
        func_val = func(doc_en)
    if ext_type == "method":
        func = textacy.get_doc_extensions("text_stats")[ext_name]["method"]
        ext_val = ext(**kwargs)
        func_val = func(doc_en, **kwargs)
    assert ext_val == func_val
    textacy.remove_doc_extensions("text_stats")
