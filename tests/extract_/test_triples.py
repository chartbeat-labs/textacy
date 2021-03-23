import re

import pytest

from textacy import load_spacy_lang
import textacy.extract_.triples


@pytest.fixture(scope="module")
def spacy_lang():
    return load_spacy_lang("en_core_web_sm")


@pytest.fixture(scope="module")
def sss_doc(spacy_lang):
    text = (
        "In general, Burton DeWilde loves animals, but he does not love *all* animals. "
        "Burton loves his cats Rico and Isaac; Burton loved his cat Lucy. "
        "But Burton DeWilde definitely does not love snakes, spiders, or moths. "
        "Now you know that Burton loves animals and cats in particular."
    )
    return spacy_lang(text)


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


@pytest.mark.parametrize(
    "entity, cue, fragment_len_range, exp",
    [
        (
            "Burton",
            "love",
            None,
            [
                (["Burton"], ["loves"], ["his", "cats", "Rico", "and", "Isaac"]),
                (["Burton"], ["loved"], ["his", "cat", "Lucy"]),
                (["Burton"], ["loves"], ["animals", "and", "cats"])
            ],
        ),
        (
            re.compile("Burton"),
            "love",
            None,
            [
                (["Burton"], ["loves"], ["his", "cats", "Rico", "and", "Isaac"]),
                (["Burton"], ["loved"], ["his", "cat", "Lucy"]),
                (["Burton"], ["loves"], ["animals", "and", "cats"])
            ],
        ),
        (
            "Burton( DeWilde)?",
            "love",
            None,
            [
                (["Burton", "DeWilde"], ["loves"], ["animals"]),
                (["Burton"], ["loves"], ["his", "cats", "Rico", "and", "Isaac"]),
                (["Burton"], ["loved"], ["his", "cat", "Lucy"]),
                (
                    ["Burton", "DeWilde"],
                    ["does", "not", "love"],
                    ["snakes", ",", "spiders", ",", "or", "moths"]
                ),
                (["Burton"], ["loves"], ["animals", "and", "cats"])
            ],
        ),
        (
            "Burton",
            "love",
            (None, 4),
            [
                (["Burton"], ["loved"], ["his", "cat", "Lucy"]),
                (["Burton"], ["loves"], ["animals", "and", "cats"])
            ],
        ),
        (
            "Burton",
            "love",
            (4, 6),
            [(["Burton"], ["loves"], ["his", "cats", "Rico", "and", "Isaac"])]
        ),
        ("Burton", "hate", None, []),
    ],
)
def test_semistructured_statements(sss_doc, entity, cue, fragment_len_range, exp):
    obs = textacy.extract_.triples.semistructured_statements(
        sss_doc, entity=entity, cue=cue, fragment_len_range=fragment_len_range
    )
    obs_text = [
        (
            [tok.text for tok in e],
            [tok.text for tok in c],
            [tok.text for tok in f]
        )
        for e, c, f in obs
    ]
    assert obs_text == exp
