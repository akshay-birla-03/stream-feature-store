.PHONY: install lint fmt test demo serve clean

install:
	pip install --break-system-packages -e ".[dev]"

lint:
	ruff check src tests

fmt:
	ruff check --fix src tests

test:
	pytest -q

demo:
	featurestore --n-users 5 --n-events 300 --n-labels 8

serve:
	uvicorn featurestore.api:app --host 0.0.0.0 --port 8000

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ src/*.egg-info build dist
