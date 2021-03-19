from functools import partial

import pytest

from textacy import preprocessing


@pytest.mark.parametrize(
    "funcs, text_in, text_out",
    [
        (
            [
                preprocessing.replace.hashtags,
                preprocessing.replace.user_handles,
                preprocessing.replace.emojis,
            ],
            "@spacy_io is OSS for industrial-strength NLP in Python developed by @explosion_ai 💥",
            "_USER_ is OSS for industrial-strength NLP in Python developed by _USER_ _EMOJI_",
        ),
        (
            [
                lambda x: x.lower(),
                preprocessing.replace.numbers,
                partial(preprocessing.remove.punctuation, only="."),
            ],
            "spaCy 3.0 introduces transformer-based pipelines for state-of-the-art performance.",
            "spacy _NUMBER_ introduces transformer-based pipelines for state-of-the-art performance ",
        ),
        # note: func order matters!
        (
            [
                preprocessing.replace.numbers,
                partial(preprocessing.remove.punctuation, only="."),
                lambda x: x.lower(),
            ],
            "spaCy 3.0 introduces transformer-based pipelines for state-of-the-art performance.",
            "spacy _number_ introduces transformer-based pipelines for state-of-the-art performance ",
        ),
    ]
)
def test_make_pipeline(funcs, text_in, text_out):
    pipeline = preprocessing.make_pipeline(*funcs)
    assert pipeline(text_in) == text_out
