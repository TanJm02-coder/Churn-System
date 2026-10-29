# Convenience tasks. Set MPLCONFIGDIR to a writable dir for matplotlib.
export MPLCONFIGDIR ?= /tmp/mpl

.PHONY: help setup data pipeline test lint format app clean

help:
	@echo "Targets:"
	@echo "  setup     create venv and install (uv)"
	@echo "  data      download the OpenML dataset"
	@echo "  pipeline  run the full modeling pipeline (reports + artifacts)"
	@echo "  test      run the pytest suite"
	@echo "  lint      ruff check + format --check"
	@echo "  format    ruff format (write)"
	@echo "  app       launch the Streamlit app"
	@echo "  clean     remove caches and generated artifacts"

setup:
	uv venv --python 3.11
	. .venv/bin/activate && uv pip install -e ".[dev]"

data:
	. .venv/bin/activate && python scripts/download_data.py

pipeline:
	. .venv/bin/activate && python scripts/run_pipeline.py

test:
	. .venv/bin/activate && python -m pytest

lint:
	. .venv/bin/activate && ruff check . && ruff format --check .

format:
	. .venv/bin/activate && ruff format .

app:
	. .venv/bin/activate && streamlit run app/streamlit_app.py

clean:
	rm -rf .pytest_cache .ruff_cache
	rm -f models/*.joblib models/*.json models/*.parquet models/*.csv
	rm -f reports/figures/*.png
