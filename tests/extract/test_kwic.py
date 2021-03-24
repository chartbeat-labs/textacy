import re

import pytest

import textacy.extract_.kwic


@pytest.fixture(scope="module")
def text():
    return (
        "Today you are You! That is truer than true! "
        "There is no one alive who is You-er than you!"
    )


class TestKeywordInContext:

    @pytest.mark.parametrize(
        "keyword, ignore_case",
        [
            ("you", True),
            ("You", False),
            ("you(-er)?", True),
            # ignore_case shouldn't affect anything when keyword is compiled regex
            (re.compile("you", flags=re.IGNORECASE), False),
            (re.compile("you(-er)", flags=re.IGNORECASE), True),
        ]
    )
    def test_keywords(self, text, keyword, ignore_case):
        obs = list(
            textacy.extract_.kwic.keyword_in_context(
                text,
                keyword,
                ignore_case=ignore_case,
            )
        )
        if isinstance(keyword, str):
            assert all(
                re.match(keyword, kw, flags=0 if ignore_case is False else re.IGNORECASE)
                for _, kw, _ in obs
            )
        else:
            assert all(re.match(keyword, kw) for _, kw, _ in obs)

    @pytest.mark.parametrize(
        "window_width, pad_context",
        [
            (10, False),
            (10, True),
            (3, True),
            (3, False),
        ]
    )
    def test_contexts(self, text, window_width, pad_context):
        obs = list(
            textacy.extract_.kwic.keyword_in_context(
                text,
                "you",
                window_width=window_width,
                pad_context=pad_context,
            )
        )
        if pad_context is False:
            assert all(
                len(pre_ctx) <= window_width and len(post_ctx) <= window_width
                for pre_ctx, _, post_ctx in obs
            )
        else:
            assert all(
                len(pre_ctx) == window_width and len(post_ctx) == window_width
                for pre_ctx, _, post_ctx in obs
            )

    @pytest.mark.parametrize(
        "keyword, ignore_case, window_width, pad_context, exp",
        [
            (
                "you",
                True,
                10,
                False,
                [
                    ('Today ', 'you', ' are You! '),
                    ('y you are ', 'You', '! That is '),
                    ('ve who is ', 'You', '-er than y'),
                    ('u-er than ', 'you', '!'),
                ],
            ),
            (
                "you",
                False,
                10,
                False,
                [
                    ('Today ', 'you', ' are You! '),
                    ('u-er than ', 'you', '!'),
                ],
            ),
            (
                "you",
                True,
                10,
                True,
                [
                    ('    Today ', 'you', ' are You! '),
                    ('y you are ', 'You', '! That is '),
                    ('ve who is ', 'You', '-er than y'),
                    ('u-er than ', 'you', '!         '),
                ],
            ),
            (
                re.compile("you", flags=re.IGNORECASE),
                True,
                10,
                False,
                [
                    ('Today ', 'you', ' are You! '),
                    ('y you are ', 'You', '! That is '),
                    ('ve who is ', 'You', '-er than y'),
                    ('u-er than ', 'you', '!'),
                ],
            ),
            (
                re.compile("you(-er)?", flags=re.IGNORECASE),
                True,
                10,
                False,
                [
                    ('Today ', 'you', ' are You! '),
                    ('y you are ', 'You', '! That is '),
                    ('ve who is ', 'You-er', ' than you!'),
                    ('u-er than ', 'you', '!'),
                ],
            ),
            (
                re.compile("you(-er)?"),
                True,  # or False, it doesn't matter
                10,
                False,
                [
                    ('Today ', 'you', ' are You! '),
                    ('u-er than ', 'you', '!'),
                ],
            ),
        ],
    )
    def test_results(self, text, keyword, ignore_case, window_width, pad_context, exp):
        obs = list(
            textacy.extract_.kwic.keyword_in_context(
                text,
                keyword,
                ignore_case=ignore_case,
                window_width=window_width,
                pad_context=pad_context,
            )
        )
        assert obs == exp
