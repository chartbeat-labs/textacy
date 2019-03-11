# contributing to textacy

Thanks for your interest in contributing! This project is maintained by [@bdewilde](https://github.com/bdewilde), and he's always glad to have help. Here's a set of guidelines to help get you started.

## have a question?

First, check around for existing resources. Is your question answered in [the docs](https://chartbeat-labs.github.io/textacy), or has it been answered somewhere else, like [StackOverflow](https://stackoverflow.com/search?q=textacy)? If not, consider your question: Is it of general interest and probably *should* be addressed in the docs? Please submit an issue that explains the question in context and where you see it fitting into the documentation. If, instead, it's a specific question about your code, StackOverflow may be a better forum for finding answers. Tag your post with `textacy` and `python` so that others may find it more easily.

## found a bug?

Search [GitHub issues](https://github.com/chartbeat-labs/textacy/issues) to see if it's already been reported. If you find an open issue that seems like the thing you're experiencing, please add a comment there; if you find a relevant closed issue but aren't satisfied with its resolution, open a new issue and link back to it in the body of your message; if you're the first to report this bug, please submit an issue explaining the problem and a minimal code example showing how to reproduce it.

## want to make some changes?

If you've found a bug *and* know how to fix it, please submit a pull request with the necessary changes — and be sure to link to any related issues, if they exist. If you'd like to suggest new functionality or other enhancements, please open an issue that briefly outlines the motivation for and scope of your proposal, giving the maintainer and other users a chance to weigh in. If there's support for the idea and you'd like to implement it yourself, please follow the code conventions and submit your changes via pull request.

## opening an issue

Use an appropriate template (if available) when [creating your issue](https://github.com/chartbeat-labs/textacy/issues/new/choose), and fill it out completely. Be sure to include a clear and descriptive title, and sufficient information for someone else to understand and reproduce the problem. A minimal code sample and/or executable test case demonstrating expected behavior is particularly helpful for diagnosing and resolving bugs. If you need to include a lot of code or a long traceback, you can wrap the content in `<details>` and `</details>` tags to collapse the content, making it easier to read and follow.

## opening a pull request

TODO

## conventions

### python

- Adhere to [PEP 8 style](https://www.python.org/dev/peps/pep-0008/) as much as is reasonable. In particular, try to keep lines to 90 characters or less; indent levels with four spaces; don't include trailing trailing whitespace, and include vertical whitespace only sparingly; and prefer double-quotes over single-quotes for string literals.
- Write code that's compatible with both Python 2.7 and Python 3.5+. (For now...) Begin each module with `from __future__ import absolute_import, division, print_function, unicode_literals` to avoid potential differences in behavior between Python versions. Additional logic that deals specifically with 2/3 compatibility should go in `textacy.compat`.
- When naming objects, strive to be both descriptive *and* brief in a way that reflects usage rather than, say, data type. Function names should be all lowercase, with words separated by underscores, and often including an action verb: `normalize_whitespace()`, `read_csv()`, `get_term_freqs()`, and so on. Objects pulled in directly from `spacy` usually have names prepended by `spacy_`, e.g. `spacy_doc` or `spacy_vocab`.

### git commits

- Strive for atomic commits. Basically, all the changes bundled into a given commit should relate to a single context, so if you're fixing multiple bugs or adding multiple features, be sure to chunk up the work into separate commits. A good rule of thumb: If you can't summarize the commit in 72 characters or less, you may be including too many changes at once.
- Write messages in the present tense (`"Add tests for..."`, not `"Added tests for..."`) and imperative mood (`"Refactor class..."`, not `"Refactors class..."`). Capitalize the first (subject) line, and don't end it with a period (`"Fix typo in Corpus docstring"`, not `"fix typo in Corpus docstring."`). Limit all lines to 72 character or less — note that git doesn't do this for you!
- Add a body to the message, if needed, to further explain the *what* and *why* of the changes, but not the *how*. Be sure to separate the body from the subject with a blank line. Reference relevant issues using a hashtag plus the number (e.g. `"#212"` will link to [this issue](https://github.com/chartbeat-labs/textacy/issues/212)).

### tests

- `textacy` uses `pytest` for testing; please refer to [their docs](https://docs.pytest.org) for details. Tests live under the top-level `tests/` directory; all tests for a given module of code are found in a like-named file prepended by `test_`, e.g. `test_preprocess.py` contains tests for the functions in `preprocess.py`.
- The current state and coverage of tests in `textacy` is, shall we say, *mixed*, so any contributions toward adding/improving tests are most welcome! In general, we'd like to have a test confirm expected behavior any time existing functionality changes or new functionality is added.

### documentation

- `textacy` uses `sphinx` for building documentation; please refer to [their docs](https://www.sphinx-doc.org) for details. Stand-alone doc files live under the top-level `docs/` directory and are written in [reStructured Text](http://docutils.sourceforge.net/docs/user/rst/quickref.html) format.
- In-code docstrings follow [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) — see some examples [here](https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html#example-google). Modules should have brief, descriptive docstrings; classes should have top-level docstrings that include usage examples; public functions and methods should have docstrings that cover the basics (a brief summary line and, if applicable, `Args` and `Returns` sections). Docstrings get incorporated into the main docs via `sphinx-autodoc` and the `api_reference.rst` file; add new modules into the API Reference, as needed.
