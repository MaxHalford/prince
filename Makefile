# Makefile

#  Configuration  #############################################################

PROJECT    := $(shell basename $(PWD))
PACKAGE    := "prince"
BUILD_TIME := $(shell date +%FT%T%z)

###############################################################################


all: install

install:
	pip install -r setup/requirements.txt
	python setup.py install

install.hack:
	pip install -r setup/dev-requirements.txt

lint:
	pylint --reports no $(PACKAGE) tests/

test: warn_missing_linters
	py.test --verbose --cov=$(PACKAGE) tests/

present_pylint=$(shell which pylint)
present_pytest=$(shell which py.test)
warn_missing_linters:
	@test -n "$(present_pylint)" || echo "WARNING: pylint not installed."
	@test -n "$(present_pytest)" || echo "WARNING: py.test not installed."

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
