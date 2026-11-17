# Deployment

## Local

```bash
pip install -r requirements-dev.txt
python scripts/download_data.py
python scripts/train_all.py --reuse-tuning
streamlit run app.py
```

The app opens on <http://localhost:8501>.

## Streamlit Community Cloud

1. **Push to GitHub.** The two model artefacts must be committed:

   ```
   models/voting_regressor.joblib   25.5 MB
   models/xgboost.joblib             1.6 MB
   models/metadata.json
   ```

   Both are under GitHub's 100 MB file limit, so no Git LFS is needed. The
   `.gitignore` ignores `models/*.joblib` and then re-includes exactly these two —
   don't add a blanket `!models/*.joblib`, or every experiment you ever run gets
   committed.

2. **Create the app** at <https://share.streamlit.io>:
   - Repository: your fork
   - Branch: `main`
   - Main file path: `app.py`
   - Python version: **3.12** (also pinned in `runtime.txt`)

3. **No secrets are required.** The app reads only committed artefacts.

## Why the requirements are split

| File | Audience | Contents |
|---|---|---|
| `requirements.txt` | Streamlit Cloud | 7 packages, ~60 MB installed |
| `requirements-dev.txt` | training | + TensorFlow, Optuna, keras-tuner, matplotlib |

Streamlit's free tier gives **1 GB RAM**. TensorFlow alone is several hundred
megabytes of import-time footprint, and the MLP it powers lost the benchmark
(173.53 MAE vs 148.97). It is a training artefact, not a serving dependency, so
it is not in the deploy image.

## Why `xgboost-cpu` and not `xgboost`

Identical version (3.4.1), identical `import xgboost`, identical pickle format.
The difference is that the default Linux `xgboost` wheel pulls in the `nvidia-*`
CUDA runtime — several hundred extra megabytes that a GPU-less host will never
use. The `xgboost-cpu` distribution is the same library compiled without them.

Local training uses the full wheel (`requirements-dev.txt`) because a developer
machine may have a GPU; the deployed app never does.

## Things that break Streamlit deployments, and how this repo avoids them

**1. CUDA in the pickle.**
XGBoost models store their `device` parameter. A booster saved with
`device="cuda"` fails on a CPU host. `models.make_xgboost()` defaults to
`device="cpu"`, so every saved artefact is host-independent.

**2. Version drift on unpickle.**
`joblib.load` does not check versions; you get either a `UserWarning` and subtly
wrong predictions, or an `AttributeError` deep inside scikit-learn. Both
requirements files pin exact versions matching the environment that produced the
artefacts.

**3. A green health check masking a broken script.**
`/_stcore/health` returns 200 as soon as the server binds a port — it says
nothing about whether your script ran. This repo caught a real instance of this:
`plotly` was missing while the health check was green.

`tests/test_app.py` fixes this by running the script through
`streamlit.testing.v1.AppTest`, which surfaces exceptions. It runs in CI.

**4. Artefact size and cold start.**
The 70-tree bagging member dominates the 25.5 MB pickle. If cold starts matter,
switch the sidebar to `xgboost` (1.6 MB) — its validation MAE is 150.06 vs
148.97, a difference smaller than the seed noise.

**5. Memory at load.**
A 25.5 MB compressed ensemble expands to roughly 200–250 MB in RAM. That fits
comfortably in 1 GB alongside Streamlit, but it is the reason the MLP is not
served.

## Verifying before you push

```bash
python -m pytest -q          # 17 tests: 13 leakage + 4 app
python -m py_compile app.py
streamlit run app.py         # click through all five tabs once
```

The app is read-only over `reports/` and `models/`. If a tab is empty, the
artefact behind it is missing — run `scripts/train_all.py --reuse-tuning` and
`scripts/make_report.py`.
