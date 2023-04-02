.PHONY: clean-build clean-py clean-test clean check-tests check-lint check-types check build download

clean-build:
	rm -rf dist build .egg .eggs **/*.egg-info

clean-py:
	find . -type f -name '*.py[co]' -delete
	find . -type d -name "__pycache__" -delete

clean-test:
	rm -f .coverage
	rm -f .coverage.*

clean: clean-build clean-py clean-test

check-tests: clean-test
	python -m pytest tests --verbose --cov=textacy --cov-report=term-missing

check-lint:
	python -m black --diff src
	python -m isort --diff src
	python -m ruff check src

check-types:
	python -m mypy src

check: check-tests check-lint check-types

build: clean-build
	python -m build --sdist --wheel

download:
	python -m spacy download en_core_web_sm
	python -m spacy download es_core_news_sm
	python -m spacy validate
	python -m textacy download capitol_words
	python -m textacy download lang_identifier --version 3.0
