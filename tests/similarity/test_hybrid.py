import pytest

from textacy.similarity import hybrid


@pytest.mark.parametrize(
    "s1, s2, exp",
    [
        ("abcd", "abcd", 1.0),
        ("abdc", "wxyz", 0.0),
        ("abcd", "", 0.0),
        ("", "", 0.0),
        ("abcd", "ABCD", 1.0),
        ("abcd", "bacd", 0.5),
        ("abcd", "abXcd", 0.8),
        ("abcd", "abcdefgh", 0.5),
        ("abcdefgh", "abcd", 0.5),
        (list("abcd"), list("abcd"), 1.0),
        (list("abcd"), list("ABCD"), 1.0),
        ("abcd", "abcd efgh", 0.5),
        (["abcd"], ["abcd", "efgh"], 0.5),
        (["abcd", "wxyz"], ["abcd", "efgh"], 0.5),
    ]
)
def test_token_sort_ratio(s1, s2, exp):
    obs = hybrid.token_sort_ratio(s1, s2)
    assert isinstance(obs, float)
    assert 0.0 <= obs <= 1.0
    assert obs == pytest.approx(exp, rel=0.01)


@pytest.mark.parametrize(
    "s1, s2, exp",
    [
        (["abcd", "efgh", "ijkl"], ["abcd", "efgh", "ijkl"], 1.0),
        (["abcd", "efgh", "ijkl"], ["opqr", "stuv", "wxyz"], 0.0),
        (["abcd", "efgh", "ijkl"], [], 0.0),
        ([], [], 0.0),
        (["abcd", "efgh", "ijkl"], ["Abcd", "efgh", "ijkl"], 0.9166),
        (["abcd", "efgh", "ijkl"], ["Abcd", "Efgh", "Ijkl"], 0.75),
        (["abcd", "efgh", "ijkl"], ["abcd", "efgh"], 0.8333),
        (["abcd", "efgh"], ["abcd", "efgh", "ijkl"], 0.8333),
        (["abcd", "efgh", "ijkl"], ["ABCD", "EFGH", "IJKL"], 0.0),
    ]
)
def test_monge_elkan(s1, s2, exp):
    obs = hybrid.monge_elkan(s1, s2)
    assert isinstance(obs, float)
    assert 0.0 <= obs <= 1.0
    assert obs == pytest.approx(exp, rel=0.01)
