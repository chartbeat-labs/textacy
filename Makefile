.PHONY: clean-build clean-py clean-test clean test lint check-types

clean-build:
	rm -rf dist build .egg .eggs **/*.egg-info

clean-py:
	find . -type f -name '*.py[co]' -delete
	find . -type d -name "__pycache__" -delete

clean-test:
	rm -f .coverage
	rm -f .coverage.*

clean: clean-build clean-py clean-test

build: clean-build
	python -m build --sdist --wheel

test: clean-test
	python -m pytest tests -v --cov=textacy --cov-report=term-missing

lint:
	python -m flake8 src

mypy:
	python -m mypy src

check: test lint mypy

download:
	python -m spacy download en_core_web_sm
	python -m spacy download es_core_news_sm
	python -m spacy validate
	python -m textacy download capitol_words
	python -m textacy download lang_identifier --version 2.0
