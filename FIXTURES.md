# Public fixtures

The public demonstration uses three frozen artefacts from the accepted ESWA
cross-architecture workflow, plus the Decision Transformer `config.json`
required by `from_pretrained`.

| Logical ID | Figshare filename | SHA-256 | Cache destination | Loader |
| --- | --- | --- | --- | --- |
| `max_quadratic_utility_TI_True.weights` | `max_quadratic_utility_TI_True.safetensors` | `439A4B08E6A76C7D97935B61F4E800CFB5DD971218B9EAA7BF7B3906DE67E95E` | `dt/max_quadratic_utility_TI_True/model.safetensors` | `models.model.TrainableDT.from_pretrained` |
| `max_quadratic_utility_TI_True.config` | `max_quadratic_utility_TI_True.config.json` | `51F07A5A6AF0A20CAC4A08688BD5E2D164D36D50DC80AD86847EADAD5990A946` | `dt/max_quadratic_utility_TI_True/config.json` | Hugging Face `from_pretrained` |
| `PPO_16_1_TI_True.weights` | `PPO_16_1_TI_True.best_model.zip` | `8CBA96DAFE397C916534B252EC97BD143C279DD3327918D3C7B76FA9D0F80690` | `rl/PPO_16_1_TI_True/best_model.zip` | `stable_baselines3.PPO.load` |
| `aligned_market_data` | `aligned_market_data.h5` | `BA6C0848AFFA06C64CC641477A590DAE4339CC6A1D13B7FEFD868BC2BED92B6E` | `data/data.h5` | `ablation.cross_architecture.data.load_market_data` |

The machine-readable contract is `fixtures/manifest.yaml`.

## Figshare

- Article ID: `33331437`
- DOI: `10.6084/m9.figshare.33331437`
- Fixture license: **CC BY 4.0**

`decision-logic-audit fetch-fixtures` queries the public Figshare article API
and matches files by the exact `figshare_filename` values above. If the
published record uses different names, update the manifest to match; do not
rename bytes locally after download except into the cache layout.

If the article is not yet public, stage files with:

```text
decision-logic-audit fetch-fixtures --from-dir PATH
```

placing files that already have the manifest filenames.

## Cache behaviour

Default cache root: `~/.cache/decision-logic-audit/`
(`%USERPROFILE%\.cache\decision-logic-audit\` on Windows).

Override: `DECISION_LOGIC_AUDIT_CACHE`.

Layout after a successful fetch:

```text
v1/
  dt/max_quadratic_utility_TI_True/config.json
  dt/max_quadratic_utility_TI_True/model.safetensors
  rl/PPO_16_1_TI_True/best_model.zip
  data/data.h5
```

SHA-256 is verified after every download or copy. A mismatch is treated as
corruption: the file is not kept for use. `--force` re-fetches.

Sanitized `metadata.json` files (instrument lists, `env_state`, lookback,
target return, PPO algorithm fields) ship with the software under
`fixtures/metadata/`. They omit researcher home-directory paths from the
original training metadata. They are not a substitute for the weight files.

## Data provenance

`aligned_market_data.h5` is a **derived / reworked research object**. It was
built from publicly accessible API observations, then daily-aggregated,
aligned across instruments, and transformed with technical indicators.

The authors claim authorship of that derived research object, not ownership
of third-party APIs or of the upstream raw observations.

## License distinction

- Python source in this repository: **GPL-3.0-only**
- Fixture files listed above: **CC BY 4.0**
