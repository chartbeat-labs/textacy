import pytest

import textacy
from textacy.text_stats import diversity


@pytest.fixture(scope="module")
def ttr_doc():
    text = (
        "Mary had a little lamb, its fleece was white as snow. "
        "And everywhere that Mary went, the lamb was sure to go."
    )
    return textacy.make_spacy_doc(text, lang="en_core_web_sm")


@pytest.fixture(scope="module")
def mtld_doc():
    text = (
        "Burton loves cats, and cats love Burton. "
        "Burton loves dogs, not as much as cats, but dogs do not love Burton very much."
    )
    return textacy.make_spacy_doc(text, lang="en_core_web_sm")


@pytest.mark.parametrize(
    "variant, exp_val", [("standard", 0.864), ("root", 4.051), ("corrected", 2.864)]
)
def test_ttr(ttr_doc, variant, exp_val):
    obs_val = diversity.ttr(ttr_doc, variant=variant)
    assert obs_val == pytest.approx(exp_val, rel=1e-2)


@pytest.mark.parametrize(
    "variant, exp_val", [("herdan", 0.953), ("summer", 0.835), ("dugast", 28.304)]
)
def test_log_ttr(ttr_doc, variant, exp_val):
    obs_val = diversity.log_ttr(ttr_doc, variant=variant)
    assert obs_val == pytest.approx(exp_val, rel=1e-2)


@pytest.mark.parametrize(
    "variant, segment_size, exp_val",
    [
        ("mean", 10, 1.0),
        ("moving-avg", 10, 1.0),
        ("mean", 20, 0.85),
        ("moving-avg", 20, 0.883),
    ],
)
def test_segmented_ttr(ttr_doc, variant, segment_size, exp_val):
    obs_val = diversity.segmented_ttr(
        ttr_doc, segment_size=segment_size, variant=variant
    )
    assert obs_val == pytest.approx(exp_val, rel=1e-2)


def test_segmented_ttr_error(ttr_doc):
    with pytest.raises(ValueError):
        _ = diversity.segmented_ttr(ttr_doc, segment_size=30)


@pytest.mark.parametrize(
    "min_ttr, exp_val",
    [(0.72, 10.181), (0.66, 14.162), (0.75, 10.062)],
)
def test_mtld(mtld_doc, min_ttr, exp_val):
    obs_val = diversity.mtld(mtld_doc, min_ttr=min_ttr)
    assert obs_val == pytest.approx(exp_val, rel=1e-2)
