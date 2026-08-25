PY=python
suite:
	$(PY) -m src.cli.suite --cfg $(CFG)
compare:
	$(PY) -m src.cli.compare --cfg $(CFG) --job $(JOB)
figs:
	$(PY) -m src.cli.figures --cfg $(CFG) --job $(JOB)
