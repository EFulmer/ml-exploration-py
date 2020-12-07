test:
	pytest -svv --cov models --cov-report term-missing tests/

format:
	black -l 79 .
