# Public fixtures

The public demonstration uses three frozen artefacts from the accepted ESWA
cross-architecture workflow, plus the Decision Transformer `config.json`
required by `from_pretrained`.

Figshare article [33331437](https://doi.org/10.6084/m9.figshare.33331437)
stores short basenames plus a folder layout. The downloader matches
**folder + filename**, then copies bytes into the cache paths below without
modifying them.

| Logical ID | Figshare path | SHA-256 | Cache destination | Loader |
| --- | --- | --- | --- | --- |
| `max_quadratic_utility_TI_True.weights` | `fixtures/decision_transformer/model.safetensors` | `439A4B08E6A76C7D97935B61F4E800CFB5DD971218B9EAA7BF7B3906DE67E95E` | `dt/max_quadratic_utility_TI_True/model.safetensors` | `models.model.TrainableDT.from_pretrained` |
| `max_quadratic_utility_TI_True.config` | `fixtures/decision_transformer/config.json` | `51F07A5A6AF0A20CAC4A08688BD5E2D164D36D50DC80AD86847EADAD5990A946` | `dt/max_quadratic_utility_TI_True/config.json` | Hugging Face `from_pretrained` |
| `PPO_16_1_TI_True.weights` | `fixtures/ppo/best_model.zip` | `8CBA96DAFE397C916534B252EC97BD143C279DD3327918D3C7B76FA9D0F80690` | `rl/PPO_16_1_TI_True/best_model.zip` | `stable_baselines3.PPO.load` |
| `aligned_market_data` | `fixtures/decision_transformer/data.h5` | `BA6C0848AFFA06C64CC641477A590DAE4339CC6A1D13B7FEFD868BC2BED92B6E` | `data/data.h5` | `ablation.cross_architecture.data.load_market_data` |

The machine-readable contract is `fixtures/manifest.yaml`.

The Figshare record also contains `fixtures/ppo/metadata.json` and a second
`data.h5` at the article root. Those files are **not** used by the public
demo. The root `data.h5` is a different object (different SHA-256) from the
frozen aligned market fixture. Sanitized model metadata ships with the
software under `fixtures/metadata/`.

## Figshare

- Article ID: `33331437`
- DOI: [`10.6084/m9.figshare.33331437`](https://doi.org/10.6084/m9.figshare.33331437)
- Fixture license: **CC BY 4.0**

`decision-logic-audit fetch-fixtures` queries the public Figshare article API,
selects files by folder + basename, verifies SHA-256, and stores them under
the cache layout. Offline staging (same names, optionally nested):

```text
decision-logic-audit fetch-fixtures --from-dir PATH
```

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

The article-root `data.h5` is unused. The frozen derived dataset is
`fixtures/decision_transformer/data.h5`. It is a **derived / reworked
research object**, built from publicly accessible API observations, then
daily-aggregated, aligned across instruments, and transformed with technical
indicators.

The authors claim authorship of that derived research object, not ownership
of third-party APIs or of the upstream raw observations.

## License distinction

- Python source in this repository: **GPL-3.0-only**
- Fixture files listed above: **CC BY 4.0**
