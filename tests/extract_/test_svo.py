import pytest

from textacy import load_spacy_lang
import textacy.extract_.svo


@pytest.fixture(scope="module")
def spacy_lang():
    return load_spacy_lang("en_core_web_sm")


@pytest.mark.parametrize(
    "text, svos_exp",
    [
        (
            "Burton loves cats.",
            [(["Burton"], ["loves"], ["cats"])],
        ),
        (
            "One of my cats was eating food.",
            [(["One"], ["was", "eating"], ["food"])],
        ),
        (
            "The first dog to arrive wins the treat.",
            [(["dog"], ["wins"], ["treat"])],
        ),
        (
            "Burton was loved by cats.",
            [(["Burton"], ["was", "loved"], ["cats"])],
        ),
        (
            "Food was eaten by my cat.",
            [(["Food"], ["was", "eaten"], ["cat"])],
        ),
        (
            "The treat was won by the first dog to arrive.",
            [(["treat"], ["was", "won"], ["dog"])],
        ),
        (
            "He and I love house cats and big dogs.",
            [(["He", "I"], ["love"], ["house", "cats", "dogs"])],
        ),
        (
            "We do love and did hate small dogs.",
            [(["We"], ["do", "love"], ["dogs"]), (["We"], ["did", "hate"], ["dogs"])],
        ),
        (
            "Rico eats food and plays fetch.",
            [(["Rico"], ["eats"], ["food"]), (["Rico"], ["plays"], ["fetch"])]
        ),
    ]
)
def test_subject_verb_object_triples(text, svos_exp, spacy_lang):
    doc = spacy_lang(text)
    svos_tok = textacy.extract_.svo.subject_verb_object_triples(doc)
    svos_obs = [
        (
            [tok.text for tok in subject],
            [tok.text for tok in verb],
            [tok.text for tok in object]
        )
        for subject, verb, object in svos_tok
    ]
    assert svos_obs == svos_exp
