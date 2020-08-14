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
	pylint src -enable W,E,F --disable C0103,C0330

mypy:
	mypy src

check: test lint mypy
