import pytest

from textacy.similarity_ import edits


@pytest.mark.parametrize(
    "str1, str2, exp",
    [
        ("abcd", "abcd", 1.0),
        ("abdc", "wxyz", 0.0),
        ("abcd", "", 0.0),
        ("", "", 0.0),
        ("abcd", "abcD", 0.75),
        ("Abcd", "abcd", 0.75),
        ("abcd", "abcdefgh", 0.5),
        ("abcdefgh", "abcd", 0.5),
    ]
)
def test_hamming(str1, str2, exp):
    obs = edits.hamming(str1, str2)
    assert isinstance(obs, float)
    assert 0.0 <= obs <= 1.0
    assert obs == pytest.approx(exp, rel=0.01)


@pytest.mark.parametrize(
    "str1, str2, exp",
    [
        ("abcd", "abcd", 1.0),
        ("abdc", "wxyz", 0.0),
        ("abcd", "", 0.0),
        ("", "", 0.0),
        ("abcd", "abcD", 0.75),
        ("Abcd", "abcd", 0.75),
        ("abcd", "bacd", 0.5),
        ("abcd", "abXcd", 0.8),
        ("abcd", "abcdefgh", 0.5),
        ("abcdefgh", "abcd", 0.5),
    ]
)
def test_levenshtein(str1, str2, exp):
    obs = edits.levenshtein(str1, str2)
    assert isinstance(obs, float)
    assert 0.0 <= obs <= 1.0
    assert obs == pytest.approx(exp, rel=0.01)
