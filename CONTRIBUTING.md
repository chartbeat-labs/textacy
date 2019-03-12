# contributing to textacy

Thanks for your interest in contributing! This project is maintained by [@bdewilde](https://github.com/bdewilde), and he's always glad to have help. Here's a set of guidelines to help get you started.

## so, you...

### have a question?

First, check around for existing resources. Is your question answered in [the docs](https://chartbeat-labs.github.io/textacy), or has it been answered somewhere else, like [StackOverflow](https://stackoverflow.com/search?q=textacy)? If not, consider your question: Is it of general interest and probably *should* be addressed in the docs? Please submit an issue that explains the question in context and where you see it fitting into the documentation. If, instead, it's a specific question about your code, StackOverflow may be a better forum for finding answers. Tag your post with `textacy` and `python` so that others may find it more easily.

### found a bug?

Search [GitHub issues](https://github.com/chartbeat-labs/textacy/issues) to see if it's already been reported. If you find an open issue that seems like the thing you're experiencing, please add a comment there; if you find a relevant closed issue but aren't satisfied with its resolution, open a new issue and link back to it in the body of your message; if you're the first to report this bug, please submit an issue explaining the problem and a minimal code example showing how to reproduce it. If you've found a bug *and* know how to fix it, please submit a pull request with the necessary changes — and be sure to follow this project's code conventions and link to any related issues, if they exist.

### want to make some changes?

If you'd like to request new or change existing functionality, first check the [GitHub issues](https://github.com/chartbeat-labs/textacy/issues) to see if it's already come up. If so, feel free to chime in on the existing issue; if not, open a new issue that briefly outlines the motivation for and scope of your proposal, giving the maintainer and other users a chance to weigh in. If there's support for the idea and you'd like to implement it yourself, please follow the code conventions and submit your changes via pull request. Otherwise, please wait patiently for someone to take up the idea and get it incorporated into the project.

## opening an issue

Use an appropriate template (if available) when [creating your issue](https://github.com/chartbeat-labs/textacy/issues/new/choose), and fill it out completely. Be sure to include a clear and descriptive title, and sufficient information for someone else to understand and reproduce the problem. A minimal code sample and/or executable test case demonstrating expected behavior is particularly helpful for diagnosing and resolving bugs — and don't forget to include details on your dev environment! If you need to include a lot of code or a long traceback, you can wrap the content in `<details>` and `</details>` tags to collapse the content, making it easier to read and follow.

## development workflow

1. **Fork the project & clone it locally:** Click the "Fork" button in the header of the [GitHub repository](https://github.com/chartbeat-labs/textacy), creating a copy of `textacy` in your GitHub account. To get a working copy on your local machine, you have to clone your fork. Click the "Clone or Download" button in the right-hand side bar, then append its output to the `git clone` command.

        $ git clone git@github.com:YOUR_USERNAME/textacy.git

1. **Create an upstream remote and sync your local copy:** Connect your local copy to the original "upstream" repository by adding it as a remote.

        $ cd textacy
        $ git remote add upstream git@github.com:chartbeat-labs/textacy.git

    You should now have two remotes: read/write-able `origin` points to your GitHub fork, and a read-only `upstream` points to the original repo. Be sure to [keep your fork in sync](https://help.github.com/en/articles/syncing-a-fork) with the original, reducing the likelihood of merge conflicts later on.

1. **Create a branch for each piece of work:** Branch off `master` for each bugfix or feature that you're working on. Give your branch a descriptive, meaningful name like `bugfix-for-issue-1234` or `improve-io-performance`, so others know at a glance what you're working on.

        $ git checkout master
        $ git pull upstream master && git push origin master
        $ git checkout -b my-descriptive-branch-name

    At this point, you may want to install your version of `textacy`. It's usually best to do this within a dedicated virtual environment; use whichever tool you're most comfortable with, such as `virtualenv`, `pyenv`, or `conda`.

        $ pyenv virtualenv 3.7.0 textacy-my-descriptive-branch-name
        $ pip install -e .

1. **Implement your changes:** Use your preferred text editor to modify the `textacy` source code. Be sure to keep your changes focused and in scope, and follow the coding conventions described below! Document your code as you write it. Run your changes against any existing tests and add new ones as needed to validate your changes; make sure you don’t accidentally break existing functionality!
1. **Push commits to your forked repository:** Group changes into atomic git commits, then push them to your `origin` repository. There's no need to wait until all changes are final before pushing — it's always good to have a backup, in case something goes wrong in your local copy.

        $ git push origin my-descriptive-branch-name

1. **Open a new Pull Request in GitHub:** When you're ready to submit your changes to the main repo, navigate to your forked repository on GitHub. Switch to your working branch then click "New pull request"; alternatively, if you recently pushed, you may see a banner at the top of the repo with a "Compare & pull request" button, which you can click on to initiate the same process. Fill out the PR template completely and clearly, confirm that code "diff" is as expected, then submit the PR.
1. **Respond to any code review feedback:** At this point, @bdewilde will review your work and either request additional changes/clarification or approve your work. There may be some necessary back-and-forth; please do your best to be responsive. If you haven’t gotten a response in a week or so, please politely nudge him in the same thread — thanks in advance for your patience!

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
