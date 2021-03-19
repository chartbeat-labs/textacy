import pytest

from textacy import preprocessing


@pytest.mark.parametrize(
    "text_in, text_out",
    [
        ("$1.00 equals 100¢.", "_CUR_1.00 equals 100_CUR_."),
        ("How much is ¥100 in £?", "How much is _CUR_100 in _CUR_?"),
        ("My password is 123$abc฿.", "My password is 123_CUR_abc_CUR_."),
    ]
)
def test_replace_currency_symbols(text_in, text_out):
    assert preprocessing.replace.currency_symbols(text_in) == text_out


@pytest.mark.parametrize(
    "text_in, text_out",
    [
        ("Reach out at username@example.com.", "Reach out at _EMAIL_."),
        ("Click here: mailto:username@example.com.", "Click here: _EMAIL_."),
    ]
)
def test_replace_emails(text_in, text_out):
    assert preprocessing.replace.emails(text_in) == text_out


@pytest.mark.parametrize(
    "text_in, text_out",
    [
        ("ugh, it's raining *again* ☔", "ugh, it's raining *again* _EMOJI_"),
        ("✌ tests are passing ✌", "_EMOJI_ tests are passing _EMOJI_"),
    ]
)
def test_replace_emojis(text_in, text_out):
    assert preprocessing.replace.emojis(text_in) == text_out


@pytest.mark.parametrize(
    "text_in, text_out",
    [
        ("like omg it's #ThrowbackThursday", "like omg it's _TAG_"),
        ("#TextacyIn4Words: \"but it's honest work\"", "_TAG_: \"but it's honest work\""),
        ("wth twitter #ican'teven #why-even-try", "wth twitter _TAG_'teven _TAG_-even-try"),
        ("www.foo.com#fragment is not a hashtag", "www.foo.com#fragment is not a hashtag"),
    ]
)
def test_replace_hashtags(text_in, text_out):
    assert preprocessing.replace.hashtags(text_in) == text_out


@pytest.mark.parametrize(
    "text_in, text_out",
    [
        (
            "I owe $1,000.99 to 123 people for 2 +1 reasons.",
            "I owe $_NUMBER_ to _NUMBER_ people for _NUMBER_ _NUMBER_ reasons.",
        ),
    ]
)
def test_replace_numbers(text_in, text_out):
    assert preprocessing.replace.numbers(text_in) == text_out


@pytest.mark.parametrize(
    "text_in, text_out",
    [
        (
            "I can be reached at 555-123-4567 through next Friday.",
            "I can be reached at _PHONE_ through next Friday.",
        ),
    ]
)
def test_replace_phone_numbers(text_in, text_out):
    assert preprocessing.replace.phone_numbers(text_in) == text_out


@pytest.mark.parametrize(
    "text_in, text_out",
    [
        (
            "I learned everything I know from www.stackoverflow.com and http://wikipedia.org/ and Mom.",
            "I learned everything I know from _URL_ and _URL_ and Mom.",
        ),
    ]
)
def test_replace_urls(text_in, text_out):
    assert preprocessing.replace.urls(text_in) == text_out


@pytest.mark.parametrize(
    "text_in, text_out",
    [
        ("like omg it's @bjdewilde", "like omg it's _USER_"),
        ("@Real_Burton_DeWilde: definitely not a bot", "_USER_: definitely not a bot"),
        ("wth twitter @b.j.dewilde", "wth twitter _USER_.j.dewilde"),
        ("foo@bar.com is not a user handle", "foo@bar.com is not a user handle"),
    ]
)
def test_replace_user_handles(text_in, text_out):
    assert preprocessing.replace.user_handles(text_in) == text_out
