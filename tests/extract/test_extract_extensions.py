import pytest

import textacy
from textacy import extract


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


def test_extensions_exist(doc):
    for name in extract.extensions.get_doc_extensions().keys():
        assert doc.has_extension(name)


@pytest.mark.parametrize(
    "ext_name, kwargs",
    [
        ("extract_words", {}),
        ("extract_ngrams", {"n": 2}),
        ("extract_entities", {}),
        ("extract_noun_chunks", {}),
        ("extract_terms", {"ngs": 2, "ents": True, "ncs": True}),
        ("extract_token_matches", {"patterns": [{"POS": "NOUN"}]}),
        ("extract_regex_matches", {"pattern": "[Mm]any"}),
        ("extract_subject_verb_object_triples", {}),
        ("extract_semistructured_statements", {"entity": "Macondo", "cue": "is"}),
        ("extract_direct_quotations", {}),
        ("extract_acronyms", {}),
        ("extract_keyword_in_context", {"keyword": "Macondo"}),
    ]
)
def test_extensions_match(doc, ext_name, kwargs):
    ext = getattr(doc._, ext_name)
    func = extract.extensions.get_doc_extensions()[ext_name]["method"]
    ext_val = ext(**kwargs)
    func_val = func(doc, **kwargs)
    if isinstance(func_val, dict):
        assert ext_val == func_val
    else:
        assert list(ext_val) == list(func_val)
