# Contributing

This is research software whose scientific algorithms are frozen relative to
an accepted study. Please keep changes in that spirit.

1. Use Python 3.12 and `pip install -e ".[dev]"`.
2. Do not change attribution, occlusion, scoring, thresholds, banding, or
   model-forward equations unless you are explicitly extending the science.
3. Run `python -m unittest tests.test_cross_architecture -v` and
   `pytest tests/test_audit_scoring tests/test_release_support.py`.
4. Public packaging lives in `pyproject.toml`. Do not reintroduce Poetry as a
   user requirement.

Contact: francisco.espiga.fernandez@alumnos.upm.es
