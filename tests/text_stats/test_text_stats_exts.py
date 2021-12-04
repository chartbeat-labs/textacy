import pytest

import textacy
from textacy import text_stats


@pytest.fixture(scope="module")
def doc():
    nlp = textacy.load_spacy_lang("en_core_web_sm")
    text = (
        "Many years later, as he faced the firing squad, Colonel Aureliano Buendía was "
        "to remember that distant afternoon when his father took him to discover ice. "
        "At that time Macondo was a village of twenty adobe houses, built on the bank "
        "of a river of clear water that ran along a bed of polished stones, which were "
        "white and enormous, like prehistoric eggs. The world was so recent that many "
        "things lacked names, and in order to indicate them it was necessary to point."
    )
    return nlp(text)


@pytest.mark.parametrize(
    "name", ["text_stats", "text_stats.basics", "text_stats.diversity"]
)
def test_set_get_remove_extensions(doc, name):
    textacy.set_doc_extensions(name)
    assert all(doc.has_extension(n) for n in textacy.get_doc_extensions(name).keys())
    if "." in name:
        assert not all(
            doc.has_extension(n) for n in textacy.get_doc_extensions("extract").keys()
        )
    textacy.remove_doc_extensions(name)
    assert not any(doc.has_extension(n) for n in textacy.get_doc_extensions(name).keys())


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
def test_extensions_match(doc, ext_name, ext_type, kwargs):
    textacy.set_doc_extensions("text_stats")
    ext = getattr(doc._, ext_name)
    if ext_type == "property":
        func = textacy.get_doc_extensions("text_stats")[ext_name]["getter"]
        ext_val = ext
        func_val = func(doc)
    if ext_type == "method":
        func = textacy.get_doc_extensions("text_stats")[ext_name]["method"]
        ext_val = ext(**kwargs)
        func_val = func(doc, **kwargs)
    assert ext_val == func_val
    textacy.remove_doc_extensions("text_stats")
