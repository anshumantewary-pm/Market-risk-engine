PYTHON ?= python

.PHONY: install test lint format download sample pipeline dashboard

install:  ## install the package and dev tools in editable mode
	$(PYTHON) -m pip install -e ".[dev]"

test:  ## run the test suite
	$(PYTHON) -m pytest -q

lint:  ## check style and common bugs
	$(PYTHON) -m ruff check .

format:  ## auto-fix lint issues where safe
	$(PYTHON) -m ruff check . --fix

download:  ## fetch real daily closes from Stooq into data/raw/prices.csv
	$(PYTHON) -m riskengine.download

sample:  ## run the full pipeline on SYNTHETIC data (no download needed)
	$(PYTHON) -m riskengine.pipeline --sample

pipeline:  ## run the full pipeline on downloaded data in data/raw/prices.csv
	$(PYTHON) -m riskengine.pipeline

dashboard:  ## open the Streamlit dashboard (run `make sample` or `make pipeline` first)
	$(PYTHON) -m streamlit run src/riskengine/dashboard/app.py
