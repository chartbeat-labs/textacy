# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

import pytest

from textacy import preprocessing


def test_replace_currency_symbols():
    in_outs = [
        ("$1.00 equals 100¢.", "_CUR_1.00 equals 100_CUR_."),
        ("How much is ¥100 in £?", "How much is _CUR_100 in _CUR_?"),
        ("My password is 123$abc฿.", "My password is 123_CUR_abc_CUR_."),
    ]
    for in_, out_ in in_outs:
        assert preprocessing.replace_currency_symbols(in_) == out_


def test_replace_emails():
    in_outs = [
        ("Reach out at username@example.com.", "Reach out at _EMAIL_."),
        ("Click here: mailto:username@example.com.", "Click here: _EMAIL_."),
    ]
    for in_, out_ in in_outs:
        assert preprocessing.replace_emails(in_) == out_


def test_replace_emojis():
    in_outs = [
        ("ugh, it's raining *again* ☔", "ugh, it's raining *again* _EMOJI_"),
        ("✌ tests are passing ✌", "_EMOJI_ tests are passing _EMOJI_"),
    ]
    for in_, out_ in in_outs:
        assert preprocessing.replace_emojis(in_) == out_


def test_replace_hashtags():
    in_outs = [
        ("like omg it's #ThrowbackThursday", "like omg it's _TAG_"),
        ("#TextacyIn4Words: \"but it's honest work\"", "_TAG_: \"but it's honest work\""),
        ("wth twitter #ican'teven #why-even-try", "wth twitter _TAG_'teven _TAG_-even-try"),
        ("www.foo.com#fragment is not a hashtag", "www.foo.com#fragment is not a hashtag"),
    ]
    for in_, out_ in in_outs:
        assert preprocessing.replace_hashtags(in_) == out_


def test_replace_numbers():
    text = "I owe $1,000.99 to 123 people for 2 +1 reasons."
    proc_text = "I owe $_NUM_ to _NUM_ people for _NUM_ _NUM_ reasons."
    assert preprocessing.replace_numbers(text, "_NUM_") == proc_text


def test_replace_phone_numbers():
    text = "I can be reached at 555-123-4567 through next Friday."
    proc_text = "I can be reached at _PHONE_ through next Friday."
    assert preprocessing.replace_phone_numbers(text, "_PHONE_") == proc_text


def test_replace_urls():
    text = "I learned everything I know from www.stackoverflow.com and http://wikipedia.org/ and Mom."
    proc_text = "I learned everything I know from _URL_ and _URL_ and Mom."
    assert preprocessing.replace_urls(text, "_URL_") == proc_text


def test_replace_user_handles():
    in_outs = [
        ("like omg it's @bjdewilde", "like omg it's _USER_"),
        ("@Real_Burton_DeWilde: definitely not a bot", "_USER_: definitely not a bot"),
        ("wth twitter @b.j.dewilde", "wth twitter _USER_.j.dewilde"),
        ("foo@bar.com is not a user handle", "foo@bar.com is not a user handle"),
    ]
    for in_, out_ in in_outs:
        assert preprocessing.replace_user_handles(in_) == out_
