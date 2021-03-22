import pytest

from textacy import load_spacy_lang
import textacy.extract_.triples


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
        (
            "What the cat did makes sense.",
            [
                (["cat"], ["did"], ["What"]),
                (["What", "the", "cat", "did"], ["makes"], ["sense"]),
            ],
        ),
        (
            "What the cat wanted was eaten by the dog.",
            [
                (["cat"], ["wanted"], ["What"]),
                (["What", "the", "cat", "wanted"], ["was", "eaten"], ["dog"]),
            ],
        ),
        (
            "Burton and Nick do not expect to adopt another cat.",
            [
                (
                    ["Burton", "Nick"],
                    ["do", "not", "expect"],
                    ["to", "adopt", "another", "cat"],
                )
            ],
        ),
        (
            "She and her friend did find, sell, and throw sea shells by the sea shore.",
            [
                (["She", "friend"], ["did", "find"], ["sea", "shells"]),
                (["She", "friend"], ["sell"], ["sea", "shells"]),
                (["She", "friend"], ["throw"], ["sea", "shells"]),
            ],
        ),
    ]
)
def test_subject_verb_object_triples(text, svos_exp, spacy_lang):
    doc = spacy_lang(text)
    svos_tok = textacy.extract_.triples.subject_verb_object_triples(doc)
    svos_obs = [
        (
            [tok.text for tok in subject],
            [tok.text for tok in verb],
            [tok.text for tok in object]
        )
        for subject, verb, object in svos_tok
    ]
    assert svos_obs == svos_exp
