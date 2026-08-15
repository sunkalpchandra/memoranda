PY := .venv/bin/python

.PHONY: setup test lint images trials units features all

setup:
	python3 -m venv --system-site-packages .venv
	.venv/bin/pip install -e ".[dev]"

test:
	$(PY) -m pytest

lint:
	.venv/bin/ruff check conceptlens scripts tests

images:
	$(PY) scripts/01_extract_images.py

trials:
	$(PY) scripts/02_extract_trials.py

units:
	$(PY) scripts/03_extract_units.py

features:
	$(PY) scripts/10_extract_features.py

all: images trials units features
