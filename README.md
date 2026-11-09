# decision-logic-audit

Research software that audits the **decision logic** of frozen portfolio-allocation
policies. The public distribution name is `decision-logic-audit`. Scientific
import names (`ablation`, `audit_scoring`, `models`, and related packages) are
unchanged.

Version **1.0.0**. License: **GPL-3.0-only**.

## Purpose

The software measures how a frozen policy uses information when it maps an
observation window to a portfolio weight vector. It does not train new policies
and it does not recommend trades.

## Relationship to the accepted ESWA paper

This repository is the reusable software distribution extracted from the accepted
Expert Systems with Applications article:

> A Diagnostic Framework for Auditing Decision Logic in Transformer-Based
> Portfolio Allocation

The full experimental archive for that study is separate. Running the public
demo here is a bounded illustration of the same algorithms, not a reproduction
of the paper's reported numerical tables.

## What the software audits

Four complementary views of a frozen policy:

- **Contribution.** Which feature families the policy relies on (attribution
  ranks, sign stability).
- **Necessity.** Whether those families are required for the observed action
  (occlusion, deletion/insertion).
- **Structure.** How organised the internal representation is (effective rank,
  linear probes, neighbourhood stability).
- **Explanation-validity overlays.** Checks that the explanations themselves
  behave coherently (baselines, randomisation, permutation, faithfulness).

Study-specific screening thresholds and bands are operational rules for that
manuscript. They are not universal standards.

## Supported demonstration

The public demo loads three frozen artefacts:

1. Decision Transformer family `max_quadratic_utility_TI_True`
2. PPO family `PPO_16_1_TI_True`
3. The derived aligned market-data HDF5 used by the existing
   cross-architecture workflow

Those files are hosted on Figshare under CC BY 4.0
(DOI [`10.6084/m9.figshare.33331437`](https://doi.org/10.6084/m9.figshare.33331437)).
Source code is GPL-3.0-only.

The demo calls the existing scientific runner
`ablation.cross_architecture.run` with `--pilot` (reduced sample size only).
It is a smoke/demo run, not a paper reproduction.

## Requirements

- Python **>=3.12,<3.13**
- A CPU-capable install is the supported default. A GPU PyTorch build may be
  installed separately if desired; CUDA is not a package requirement.

## Installation with venv and pip

### Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install .
```

### Linux / macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install .
```

Development extras (tests and build tools):

```bash
pip install -e ".[dev]"
```

Optional extras: `viz` (UMAP projections) and `extra` (SHAP / XGBoost /
PyPortfolioOpt helpers that are not required by the public demo).

## Fetching Figshare fixtures

```powershell
decision-logic-audit fetch-fixtures
```

The command queries the public Figshare article API, selects files by folder
and basename, verifies SHA-256, and stores them under:

```text
%USERPROFILE%\.cache\decision-logic-audit\v1\   (Windows)
~/.cache/decision-logic-audit/v1/              (Linux/macOS)
```

Override the cache root with `DECISION_LOGIC_AUDIT_CACHE`.

Offline staging from a directory that uses the Figshare names (nested folders
or a flat copy of the four demo files):

```powershell
decision-logic-audit fetch-fixtures --from-dir PATH\to\figshare_files
```

See [FIXTURES.md](FIXTURES.md) for hashes, roles, and data provenance.

## Running the pilot demo

```powershell
decision-logic-audit fetch-fixtures
decision-logic-audit demo
```

Outputs are written to `outputs/demo/` (override with `--output-dir`).
The demo is labelled a smoke/pilot run. It does not retune thresholds or
rerun the full ESWA study.

Equivalent scientific command after fixtures are cached:

```powershell
python -m ablation.cross_architecture.run --pilot --skip-tests --device cpu
```

(with `--dt-model-dir`, `--rl-model-dir`, and `--data-path` pointing at the
materialised cache layout).

## Main reusable modules / entry points

| Entry | Role |
| --- | --- |
| `decision-logic-audit` | Fetch fixtures, run demo, print environment diagnostics |
| `python -m ablation.cross_architecture.run` | Cross-architecture diagnostic comparison |
| `python -m audit_scoring.cli` | Batch A contribution–necessity–structure screening |
| `python -m audit_scoring.holdout_cli` | Batch B temporally separated holdout assessment |
| `python -m audit_scoring.validity_cli` | Batch C validity overlay |
| `python -m audit.generate_batch_a_diagnostics` | Preferred Batch A diagnostic generator |
| `scripts/generate_batch_a_diagnostics.py` | Retained standalone duplicate of that generator |

Importable scientific packages include `ablation`, `audit`, `audit_scoring`,
`models`, `regime_representation`, `rl_factory_shim`, `financial_alignment`,
and `model_comparison`.

## Documentation

[Adding another model architecture and adapting to other sequential domains](docs/ADDING_MODEL_ARCHITECTURE.md)

## Output locations

- Public demo: `outputs/demo/`
- Fixture cache: `~/.cache/decision-logic-audit/v1/`
- Longer Batch A–C runs: `results/` as configured on the CLI

## Scope and limitations

This is **research software**.

- It is not an investment recommendation.
- It does not predict future returns.
- It does not certify causal claims.
- It does not authorise deployment.
- Study-specific screening thresholds should not be treated as universal.

## Reproducibility relationship

The accepted ESWA paper has a separate experimental archive (full model
families, diagnostic pickles, and manuscript tables). This repository is the
portable, pip-installable software distribution plus a three-file public
demonstration.

## License

- **Source code:** GNU General Public License v3.0 only (`GPL-3.0-only`)
- **Public fixture files:** Creative Commons Attribution 4.0 (`CC BY 4.0`)

The two licenses apply to different objects. Do not treat model weights or the
HDF5 file as GPL-covered source, and do not treat the Python package as CC BY.

## Citation

Please cite the accepted ESWA article when using this software, and the
software version via `CITATION.cff`.

Espiga-Fernández, F., Ordieres-Meré, J., & García-Sánchez, Á. *A Diagnostic
Framework for Auditing Decision Logic in Transformer-Based Portfolio
Allocation*. Expert Systems with Applications (accepted).

## Support / contact

Francisco Espiga-Fernández  
Industrial Engineering, Business Administration and Statistics Department,
ETSII, Universidad Politécnica de Madrid  
<francisco.espiga.fernandez@alumnos.upm.es>
