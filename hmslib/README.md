# hmslib

Statistics and fault detection & diagnosis for liquid rocket engines.

Built for an **offline Windows machine**: no installation step, and no
dependency beyond what is certainly already there — numpy, scipy, pandas,
scikit-learn, matplotlib (torch only for the future `hmslib.nn`). No seaborn,
no pyyaml, no pyarrow.

## Install

Copy the `hmslib` folder next to your scripts or notebooks, then

```python
import sys; sys.path.insert(0, r"C:\path\to\the\folder\containing\hmslib")
import hmslib as hm
hm.check_env()          # prints every version found and flags anything risky
```

`check_env()` is worth running first in every session: the code targets
Python 3.9+ and avoids the APIs removed or renamed in recent numpy, pandas,
scikit-learn and matplotlib releases, but this tells you what you actually have.

## Five minute workflow

```python
import hmslib as hm

# 1. discover the files and write a manifest -- then open it and fix it
hm.io.scan_folder("data/", write="manifest.json")

# 2. load; every column-role inference is printed, none is silent
ds = hm.Dataset.from_manifest("manifest.json")
op = ds["OP_100"]

# 3. is the data usable at all?
op.quality("nominal").describe()

# 4. one PDF per operating point with everything worth looking at
hm.quicklook(ds, out="reports/")

# 5. a detector per operating point
det  = hm.Mahalanobis(cov="ledoit_wolf", threshold="empirical", alpha=1e-3)
bank = hm.ModelBank.fit(ds, det)
bank.describe()
bank.save("models/")
```

`templates/01_quicklook_and_detection.ipynb` walks through the same path with
commentary, and runs on synthetic data when no real data are available yet.

## Data layout

One CSV with the nominal Monte Carlo cloud and one with the failure runs, per
operating point. File and column names are unknown in advance, so:

* `scan_folder` *proposes* a pairing (by directory, or by file-name tokens) and
  writes `manifest.json`; anything it could not pair is listed under
  `unmatched` rather than dropped;
* the manifest is the single source of truth afterwards and is meant to be
  edited by hand — operating point names, `columns.label`, `columns.intensity`,
  `columns.exclude`, `columns.sensors`;
* `infer_schema` decides column roles mainly by comparing the failure table
  against the nominal one: label and intensity exist only in the former.
  Use `"none"` in the manifest to state that a column does not exist.

Sensors disabled at some operating points are fine: models are per-OP.
`ds.common_sensors()` and `ds.sensor_availability()` are there for the cases
where you need to compare across operating points.

## Modules

| module | what it does |
|---|---|
| `compat` | `check_env`, wrappers for the unstable APIs |
| `config` | seeds, figure style |
| `schema` | column-role inference, intensity units guess |
| `io` | `scan_folder`, manifest, `Dataset`, stratified splits |
| `quality` | missing values, constants, duplicates, collinearity, effective rank |
| `preprocess` | standard/robust scaler, redundancy removal, synthetic noise |
| `viz` | histograms, correlation, spectrum, PCA, `trend_vs_intensity`, POD curves, contributions |
| `detect` | `Mahalanobis` (see below), sharing the contract in `detect.base` |
| `bank` | `ModelBank`: one model per operating point |
| `analysis` | detectability: POD curves, `i50/i90/i90_95`, sensor sensitivity |
| `report` | `quicklook` and `detection_report`, multipage PDFs |
| `synth` | synthetic generator, for the tests and for rehearsing the workflow |

## The Mahalanobis chain

The distance is trivial; everything that goes wrong lives in the inverse
covariance. On engine data the sensors are redundant by construction, so the
covariance is routinely singular. The chain is explicit at every step:

1. **structural pre-check** — constant and redundant sensors removed *before*
   estimating anything;
2. **scaling** — `'standard'` or `'robust'` (median/MAD);
3. **covariance** — `'empirical' | 'ledoit_wolf' | 'oas' | 'mcd' | 'diagonal'`;
4. **factorisation, never an explicit inverse** — Cholesky of `Sigma + ridge*I`,
   or a truncated eigendecomposition; both give a matrix `A` with
   `d^2 = ||A (x - mu)||^2`;
5. **diagnostics** — `det.diagnostics_` reports condition number, effective
   rank, ridge actually used and samples per sensor; `det.warnings_` says in
   words what is wrong;
6. **thresholds** — chi-square, Hotelling `T^2` with the finite-sample F
   correction, and an out-of-sample empirical quantile. All three are computed
   and stored in `det.thresholds_`; `det.threshold_table()` compares them.

`det.contributions(X)` decomposes `d^2` over sensors (the terms sum exactly to
the distance), which is what turns detection into isolation;
`det.top_contributors(X, k)` names the k sensors carrying most of it.

On the reference SSME tables, where two sensors are literally the same column,
the difference is not cosmetic — at a nominal 0.1% target:

| chain | sensors | components | actual FPR | detected |
|---|---|---|---|---|
| default (drop + Ledoit-Wolf + Cholesky) | 24/26 | 24 | 0.20% | 100% |
| naive (keep all + empirical covariance) | 26/26 | 26 | **4.73%** | 100% |
| truncated (eigen, 99% variance) | 26/26 | 7 | 0.13% | 99.7% |

## Detectability

The fault intensity is swept *inside* the Monte Carlo loop, so at a given
intensity the runs still scatter: detection is a random event, and "from which
intensity is the fault visible" is answered by a probability, not a number.
`hmslib.analysis` fits the POD (probability of detection) curve of each failure
class and reports

* `i50`, `i90` — intensity detected 50% / 90% of the time;
* `i90_95` — upper 95% confidence bound on `i90`, from a bootstrap over the
  runs. The conservative figure, always `>= i90`;
* the same two crossings read off the binned empirical curve (`i50_emp`,
  `i90_emp`), so a disagreement points at the model assumption rather than at
  the data.

```python
results = hm.analysis.pod_analysis(det, op, at_alpha=1e-3, n_boot=200)
hm.analysis.pod_table(results)      # hardest classes first
hm.viz.plot_pod(results)
hm.detection_report(bank, ds, out="reports/detection.pdf", at_alpha=1e-3)
```

Two things the module refuses to let you forget:

* **a POD curve holds at one false positive rate**. Lowering the threshold buys
  detection for free, so `at_alpha` re-calibrates the detector on nominal data
  before the analysis and every result records the rate it holds at;
* **an `i90` past the end of the sweep is an extrapolation**, and says so in
  `res.notes`. Same for classes that are always, or never, detected: those are
  flagged rather than fitted silently.

The parametric model is the classic log-odds form
`logit(POD) = b0 + b1·log(intensity)` (Berens, MIL-HDBK-1823), fitted by
maximum likelihood with an IRLS written out in `analysis._logistic_irls` —
scikit-learn's logistic regression regularises by default and the way to switch
that off has changed name across versions.

`analysis.sensor_sensitivity` answers the companion question: which sensor
reacts first, and how cleanly, for each class.

## Tests

```
python -m pytest tests -q
```

120 tests, all on synthetic data with known ground truth: the chi-square
behaviour of `d^2`, the false positive rate actually delivered at a requested
alpha, the exactness of the contribution decomposition, rank-deficiency
handling, MCD's resistance to contamination, save/load round trips, and the
recovery of `i50` and `i90` from a POD curve built by construction.

The strongest of these compares the fitted POD against the closed form: for a
Gaussian nominal cloud and a mean-shifting fault, `d^2` follows a non central
chi-square, so the true POD is computable exactly. The fitted `i90` lands
within 3% of it.

## Status

Implemented: data discovery and loading, quality, scaling, visualisation,
quicklook report, Mahalanobis detector, model bank, detectability and POD.

Planned next: one-class SVM and the open-set classifier with its rejection
threshold, `evaluate`, `nn` (autoencoder detector and MLP classifier).
