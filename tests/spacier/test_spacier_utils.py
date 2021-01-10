import pytest
from spacy.tokens import Doc

from textacy import load_spacy_lang
from textacy.spacier import utils


@pytest.fixture(scope="module")
def spacy_doc():
    spacy_lang = load_spacy_lang("en")
    text = """
    The unit tests aren't going well.
    I love Python, but I don't love backwards incompatibilities.
    No programmers were permanently damaged for textacy's sake.
    Thank God for Stack Overflow.
    """
    spacy_doc = spacy_lang(text.strip())
    return spacy_doc


def test_preserve_case(spacy_doc):
    results = [utils.preserve_case(tok) for tok in spacy_doc]
    assert all(isinstance(result, bool) for result in results)
    assert (
        sum(1 if result else 0 for result in results) <
        sum(1 if not result else 0 for result in results)
    )


def test_get_normalized_text(spacy_doc):
    expected = [
        "the",
        "unit",
        "test",
        "be",
        "not",
        "go",
        "well",
        ".",
        "-PRON-",
        "love",
        "Python",
        ",",
        "but",
        "-PRON-",
        "do",
        "not",
        "love",
        "backwards",
        "incompatibility",
        ".",
        "no",
        "programmer",
        "be",
        "permanently",
        "damage",
        "for",
        "textacy",
        "'s",
        "sake",
        ".",
        "thank",
        "God",
        "for",
        "Stack",
        "Overflow",
        ".",
    ]
    observed = [utils.get_normalized_text(tok) for tok in spacy_doc if not tok.is_space]
    assert observed == expected


def test_get_main_verbs_of_sent(spacy_doc):
    expected = [["going"], ["love", "love"], ["damaged"], ["Thank"]]
    observed = [
        [tok.text for tok in utils.get_main_verbs_of_sent(sent)]
        for sent in spacy_doc.sents
    ]
    for obs, exp in zip(observed, expected):
        assert obs == exp


def test_get_subjects_of_verb(spacy_doc):
    expected = [["tests"], ["I"], ["I"], ["programmers"], []]
    main_verbs = [
        tok for sent in spacy_doc.sents for tok in utils.get_main_verbs_of_sent(sent)
    ]
    observed = [
        [tok.text for tok in utils.get_subjects_of_verb(main_verb)]
        for main_verb in main_verbs
    ]
    for obs, exp in zip(observed, expected):
        assert obs == exp


def test_get_objects_of_verb(spacy_doc):
    expected = [[], ["Python"], ["incompatibilities"], ["sake"], ["God", "Overflow"]]
    main_verbs = [
        tok for sent in spacy_doc.sents for tok in utils.get_main_verbs_of_sent(sent)
    ]
    observed = [
        [tok.text for tok in utils.get_objects_of_verb(main_verb)]
        for main_verb in main_verbs
    ]
    for obs, exp in zip(observed, expected):
        assert obs == exp


def test_make_doc_from_text_chunks():
    text = "Burton forgot to add tests for this function."
    for lang in ("en", load_spacy_lang("en")):
        spacy_doc = utils.make_doc_from_text_chunks(text, lang)
        assert isinstance(spacy_doc, Doc)
        assert spacy_doc.text == text
