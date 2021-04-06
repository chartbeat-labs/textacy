import pytest

from textacy.similarity import tokens


@pytest.mark.parametrize(
    "seq1, seq2, exp",
    [
        (list("abcd"), list("abcd"), 1.0),
        (list("abcd"), list("wxyz"), 0.0),
        (list("abcd"), [], 0.0),
        ([], [], 0.0),
        (list("abcd"), list("abcD"), 0.6),
        (list("Abcd"), list("abcd"), 0.6),
        (list("abcd"), list("abcdefgh"), 0.5),
    ]
)
def test_jaccard(seq1, seq2, exp):
    obs = tokens.jaccard(seq1, seq2)
    assert isinstance(obs, float)
    assert 0.0 <= obs <= 1.0
    assert obs == pytest.approx(exp, rel=0.01)


@pytest.mark.parametrize(
    "seq1, seq2, exp",
    [
        (list("abcd"), list("abcd"), 1.0),
        (list("abcd"), list("wxyz"), 0.0),
        (list("abcd"), [], 0.0),
        ([], [], 0.0),
        (list("abcd"), list("abcD"), 0.75),
        (list("Abcd"), list("abcd"), 0.75),
        (list("abcd"), list("abcdefgh"), 0.6666),
    ]
)
def test_sorensen_dice(seq1, seq2, exp):
    obs = tokens.sorensen_dice(seq1, seq2)
    assert isinstance(obs, float)
    assert 0.0 <= obs <= 1.0
    assert obs == pytest.approx(exp, rel=0.01)


@pytest.mark.parametrize(
    "seq1, seq2, alpha, beta, exp",
    [
        (list("abcd"), list("abcd"), 1.0, 1.0, 1.0),
        (list("abcd"), list("wxyz"), 1.0, 1.0, 0.0),
        (list("abcd"), [], 1.0, 1.0, 0.0),
        ([], [], 1.0, 1.0, 0.0),
        (list("abcd"), list("abcD"), 1.0, 1.0, 0.75),
        (list("Abcd"), list("abcd"), 1.0, 1.0, 0.75),
        (list("abcd"), list("abcdefgh"), 1.0, 1.0, 1.0),
        (list("abcd"), list("abcdefgh"), 0.5, 1.0, 0.6666),
        (list("abcd"), list("abcdefgh"), 0.5, 2.0, 0.5),
    ]
)
def test_tversky(seq1, seq2, alpha, beta, exp):
    obs = tokens.tversky(seq1, seq2, alpha=alpha, beta=beta)
    assert isinstance(obs, float)
    assert 0.0 <= obs <= 1.0
    assert obs == pytest.approx(exp, rel=0.01)


@pytest.mark.parametrize(
    "seq1, seq2, alpha, beta, equiv",
    [
        (list("abcd"), list("abcD"), 0.5, 2.0, tokens.jaccard),
        (list("abcd"), list("abcdefgh"), 0.5, 2.0, tokens.jaccard),
        (list("abcd"), list("abcD"), 0.5, 1.0, tokens.sorensen_dice),
        (list("abcd"), list("abcdefgh"), 0.5, 1.0, tokens.sorensen_dice),
    ]
)
def test_tversky_equivalence(seq1, seq2, alpha, beta, equiv):
    obs = tokens.tversky(seq1, seq2, alpha=alpha, beta=beta)
    obs_equiv = equiv(seq1, seq2)
    assert obs == obs_equiv


@pytest.mark.parametrize(
    "seq1, seq2, exp",
    [
        (list("abcd"), list("abcd"), 1.0),
        (list("abcd"), list("wxyz"), 0.0),
        (list("abcd"), [], 0.0),
        ([], [], 0.0),
        (list("abcd"), list("abcD"), 0.75),
        (list("Abcd"), list("abcd"), 0.75),
        (list("abcd"), list("abcdefgh"), 0.7071),
    ]
)
def test_cosine(seq1, seq2, exp):
    obs = tokens.cosine(seq1, seq2)
    assert isinstance(obs, float)
    assert 0.0 <= obs <= 1.0
    assert obs == pytest.approx(exp, rel=0.01)


@pytest.mark.parametrize(
    "seq1, seq2, exp",
    [
        (list("abcd"), list("abcd"), 1.0),
        (list("abcd"), list("wxyz"), 0.0),
        (list("abcd"), [], 0.0),
        ([], [], 0.0),
        (list("abcd"), list("abcD"), 0.75),
        (list("Abcd"), list("abcd"), 0.75),
        (list("abcd"), list("abcdefgh"), 0.5),
        (list("abcd"), list("aabcd"), 0.8),
        (list("abcd"), list("aabcdd"), 0.6666),
        (list("abcd"), list("aabbccdd"), 0.5),
    ]
)
def test_bag(seq1, seq2, exp):
    obs = tokens.bag(seq1, seq2)
    assert isinstance(obs, float)
    assert 0.0 <= obs <= 1.0
    assert obs == pytest.approx(exp, rel=0.01)
