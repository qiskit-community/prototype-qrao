# This Makefile is provided for convenience; all it does it call tox

error:
	@echo "Please choose one of the following targets: lint, test, test-backends, testall, black, coverage, docs"
	@exit 2

test: lint
	tox -epy3

test-backends: lint
	tox -epy3 -- --run-backend-tests

coverage: lint
	tox -ecoverage

testall:
	tox -p -- --run-backend-tests

lint:
	tox -elint

black:
	tox -eblack

docs:
	tox -edocs

.PHONY: error test test-backends testall lint black docs coverage
