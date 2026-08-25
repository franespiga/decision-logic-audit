# Third-party notices

This distribution's source is licensed **GPL-3.0-only**. Runtime libraries are
pulled in by pip and keep their own licenses.

Notable dependencies used by the public scientific paths:

- PyTorch (BSD-style)
- Hugging Face `transformers`, including the Decision Transformer API (Apache-2.0)
- Captum (BSD-style)
- Stable-Baselines3 (MIT)
- Gymnasium, NumPy, SciPy, pandas, scikit-learn, statsmodels, matplotlib,
  seaborn, PyYAML, cloudpickle, h5py, PyTables, safetensors, tqdm

`src/models` implements a project-specific `TrainableDT` that subclasses the
Hugging Face Decision Transformer model/config API. It is not a verbatim
vendor drop of the Transformers source tree.

No other vendored third-party source with a conflicting license was identified
in the extraction used for this release.
