PY ?= python

.PHONY: install-dev test fetch-fixtures demo clean

install-dev:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pip check
	$(PY) -m unittest tests.test_cross_architecture -v
	$(PY) -m pytest tests/test_audit_scoring tests/test_release_support.py

fetch-fixtures:
	$(PY) -m decision_logic_audit fetch-fixtures

demo:
	$(PY) -m decision_logic_audit demo

clean:
	$(PY) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in map(pathlib.Path, ['build','dist','outputs','.pytest_cache'])]; import glob; [shutil.rmtree(p, ignore_errors=True) for p in glob.glob('*.egg-info')+glob.glob('src/*.egg-info')]"
