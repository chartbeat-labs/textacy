# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

from textacy.constants import (
    RE_ACRONYM,
    RE_URL,
    RE_EMAIL,
    RE_PHONE,
    RE_NUMBERS,
    RE_SHORT_URL,
)


GOOD_ACRONYMS = [
    "LGTM",
    "U.S.A.",
    "PEP8",
    "LGBTQQI2S",
    "TF-IDF",
    "D3",
    "3D",
    "3-D",
    "3D-TV",
    "D&D",
    "PrEP",
    "H2SO4",
    "I/O",
    "WASPs",
    "G-8",
    "A-TReC",
]
BAD_ACRONYMS = ["A", "GHz", "1a", "D o E", "Ms", "Ph.D", "3-Dim."]

GOOD_PHONES = [
    "1-722-686-3338",
    "984.744.3425",
    "(188)273-7152",
    "+1 (616) 555-3439",
    "(027)458-7382x7531",
    "727.769.2515 #64526",
    "(535) 327 1955 ext. 902",
    "1-492-748-2325-1056",
    "099 145 5237",
    "040.351.7778x63654",
    "123-4567",
]
BAD_PHONES = [
    "+36(0)4963872475",
    "2015-12-23",
    "12/23/2015",
    "(044) 664 123 45 67",
    "(12) 3456 7890",
    "01234 567890",
    "91 23 45 678",
    "12-345-67-89",
    "123,456,789",
]
PARTIAL_PHONES = ["(0123) 456 7890"]

GOOD_URLS = [
    "http://foo.com/blah_blah",
    "http://foo.com/blah_blah/",
    "http://foo.com/blah_blah_(wikipedia)",
    "http://foo.com/blah_blah_(wikipedia)_(again)",
    "http://www.example.com/wpstyle/?p=364",
    "https://www.example.com/foo/?bar=baz&inga=42&quux",
    "http://✪df.ws/123",
    "http://userid:password@example.com:8080",
    "http://userid:password@example.com:8080/",
    "http://userid@example.com",
    "http://userid@example.com/",
    "http://userid@example.com:8080",
    "http://userid@example.com:8080/",
    "http://userid:password@example.com",
    "http://userid:password@example.com/",
    "http://142.42.1.1/",
    "http://142.42.1.1:8080/",
    "http://➡.ws/䨹",
    "http://⌘.ws",
    "http://⌘.ws/",
    "http://foo.com/blah_(wikipedia)#cite-1",
    "http://foo.com/blah_(wikipedia)_blah#cite-1",
    "http://foo.com/unicode_(✪)_in_parens",
    "http://foo.com/(something)?after=parens",
    "http://☺.damowmow.com/",
    "http://code.google.com/events/#&product=browser",
    "http://j.mp",
    "ftp://foo.bar/baz",
    "http://foo.bar/?q=Test%20URL-encoded%20stuff",
    "http://مثال.إختبار",
    "http://例子.测试",
    "http://उदाहरण.परीक्षा",
    "http://-.~_!$&'()*+,;=:%40:80%2f::::::@example.com",
    "http://1337.net",
    "http://a.b-c.de",
    "http://223.255.255.254",
    "www.foo.com",
    "www3.foo.com/bar",
]
BAD_URLS = [
    "http://",
    "http://.",
    "http://..",
    "http://../",
    "http://?",
    "http://??",
    "http://??/",
    "http://#",
    "http://##",
    "http://##/",
    "//",
    "//a",
    "///a",
    "///",
    "http:///a",
    "foo.com",
    "rdar://1234",
    "h://test",
    "http:// shouldfail.com",
    ":// should fail",
    "ftps://foo.bar/",
    "http://-error-.invalid/",
    "http://a.b--c.de/",
    "http://-a.b.co",
    "http://a.b-.co",
    "http://0.0.0.0",
    "http://10.1.1.0",
    "http://10.1.1.255",
    "http://224.1.1.1",
    "http://123.123.123",
    "http://3628126748",
    "http://.www.foo.bar/",
    "http://.www.foo.bar./",
    "http://10.1.1.1",
    "foo.bar",
    "can.not.even",
]
PARTIAL_URLS = [
    "http://foo.bar/foo(bar)baz quux",
    "http://1.1.1.1.1",
    "http://foo.bar?q=Spaces should be encoded",
    "http://www.foo.bar./",
]

GOOD_SHORT_URLS = [
    "http://adf.ly/1TxZVO",
    "https://goo.gl/dmL4Gm",
    "http://chart.bt/1OLMAOm",
    "http://ow.ly/Whoc3",
    "http://tinyurl.com/oj6fudq",
    "http://tiny.cc/da1j7x",
    "https://tr.im/KI7ef",
    "http://is.gd/o46NHa",
    "http://yep.it/ywdiux",
    "http://snipurl.com/2adb7eb",
    "http://adyou.me/I4YW",
    "http://nyti.ms/1TgPKgX",
    "http://qr.net/bpfqD",
    "https://chartbeat.com/about/",
    "http://subfoo.foo.bar/abcd",
]
BAD_SHORT_URLS = [
    "ftp://foo.bar/baz",
    "http://foo.com/blah_blah?adsf",
    "https://www.example.com/foo/?bar=baz&inga=42&quux",
]

GOOD_EMAILS = [
    "prettyandsimple@example.com",
    "very.common@example.com",
    "disposable.style.email.with+symbol@example.com",
    "other.email-with-dash@example.com",
    "üñîçøðé@example.com",
    "üñîçøðé@üñîçøðé.com",
    "example@s.solutions",
    "あいうえお@example.com",
]
BAD_EMAILS = [
    "plainaddress",
    "abc.example.com",
    "A@b@c@example.com",
    "john..doe@example.com",
    "john.doe@example..com",
    "#@%^%#$@#$@#.com",
    "@example.com",
    "email.example.com",
    "“email”@example.com",
    "email@example",
    "email@-example.com",
    ".email@example.com",
    "email.@example.com",
    "email@111.222.333.44444",
    "email@[123.123.123.123]",
    "user@[IPv6:2001:db8::1]]",
]

GOOD_NUMBERS = [
    "1",
    "1,234",
    "1.234",
    "1 234",
    "1,234.56",
    "1.234,56",
    "1 234,56",
    "123,456,789",
    "123.456.789",
    "123,456.789",
    "123 456.789",
    "123 456,789",
    "-123",
    "–123",
    "+123",
    "0.123",
    ".123",
    "3.141592653589793238",
    "1,000,000,000,000.00",
]
BAD_NUMBERS = ["D3", "3D", "1st"]
PARTIAL_NUMBERS = [
    "111-111",
    "011,111,111",
    "111.",
    "01,111,111",
    "$123.45",
    "(555)123-456",
    "2015-12-24",
]


def test_good_acronym_regex():
    for item in GOOD_ACRONYMS:
        assert item == RE_ACRONYM.search(item).group()


def test_bad_acronym_regex():
    for item in BAD_ACRONYMS:
        assert RE_ACRONYM.search(item) is None


def test_good_url_regex():
    for item in GOOD_URLS:
        assert item == RE_URL.search(item).group()


def test_bad_url_regex():
    for item in BAD_URLS:
        assert RE_URL.search(item) is None


def test_partial_url_regex():
    for item in PARTIAL_URLS:
        assert item != RE_URL.search(item)


def test_good_short_url_regex():
    for item in GOOD_SHORT_URLS:
        assert item == RE_SHORT_URL.search(item).group()


def test_bad_short_url_regex():
    for item in BAD_SHORT_URLS:
        assert RE_SHORT_URL.search(item) is None


def test_good_phone_regex():
    for item in GOOD_PHONES:
        assert item == RE_PHONE.search(item).group()


def test_bad_phone_regex():
    for item in BAD_PHONES:
        assert RE_PHONE.search(item) is None


def test_partial_phone_regex():
    for item in PARTIAL_PHONES:
        assert item != RE_PHONE.search(item)


def test_good_email_regex():
    for item in GOOD_EMAILS:
        assert item == RE_EMAIL.search(item).group()


def test_bad_email_regex():
    for item in BAD_EMAILS:
        assert RE_EMAIL.search(item) is None


def test_good_number_regex():
    for item in GOOD_NUMBERS:
        assert item == RE_NUMBERS.search(item).group()


def test_bad_number_regex():
    for item in BAD_NUMBERS:
        assert RE_NUMBERS.search(item) is None


def test_partial_number_regex():
    for item in PARTIAL_NUMBERS:
        assert item != RE_NUMBERS.search(item)
