clean-py:
	find . -type f -name '*.py[co]' -delete
	find . -type d -name "__pycache__" -delete

clean-test:
	rm -f .coverage
	rm -f .coverage.*

clean: clean-py clean-test

test: clean
	python -m pytest tests -v --cov=textacy --cov-report=term-missing

lint:
	python -m flake8 src

mypy:
	python -m mypy src

check: test lint mypy
