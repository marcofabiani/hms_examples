# hms_examples

Health monitoring and fault detection for liquid rocket engines — a worked example on
the RS-25 (SSME), and `hmslib`, the small library it grew into.

Companion code and data for:

> F. Mattia, M. Fabiani, M. Fiore, F. Nasuti,
> **Data-Driven Health Monitoring and Fault Detection for the RS-25 Staged Combustion
> Liquid Rocket Engine in EcosimPro/ESPSS Framework**,
> *Space Propulsion 2026*, Bari, Italy, 18–21 May 2026.
> → [paper (PDF)](papers/SP2026_HMS.pdf)

---

## What's here

| | |
|---|---|
| [`data/`](data/) | nominal and failure datasets from an RS-25 digital twin |
| [`workflow.ipynb`](workflow.ipynb) | the minimal example: load, scale, Mahalanobis, plot |
| [`helpers.py`](helpers.py) | ~100 lines supporting the minimal example |
| [`hmslib/`](hmslib/) | the library: discovery, quality, detection, detectability |
| [`templates/`](templates/) | two notebooks walking through `hmslib` end to end |
| [`tests/`](tests/) | 120 tests, on synthetic data with known ground truth |
| [`papers/`](papers/) | the paper |

Two layers, on purpose. `workflow.ipynb` is short enough to read in one sitting and
shows the idea. `hmslib` is what you need when the data change shape, the covariance
turns out to be singular, and somebody asks how many false alarms per hour you expect.

## The data

Steady-state operating points from a digital twin of the RS-25 built in EcosimPro/ESPSS.
Each row is one converged run; there is no time axis.

- `Table_Nominal_Rid.csv` — 3000 Monte Carlo samples around the nominal operating
  point, 26 sensors
- `Table_Failures_3sigma_26Sensori.csv` — 24061 runs across 28 labelled failure
  classes (valve leakages, anomalous openings, turbine and pump efficiency losses,
  injector area reductions, cooling occlusions), same 26 sensors plus a `Failure` column

## Quick start

```bash
pip install numpy scipy pandas scikit-learn matplotlib
python -m pytest tests -q          # 120 tests, no data needed
jupyter lab workflow.ipynb         # the minimal example
```

Then, with the library:

```python
import hmslib as hm

hm.check_env()                                     # what is actually installed
hm.io.scan_folder("data/", write="manifest.json")  # proposes a nominal/failure pairing
ds = hm.Dataset.from_manifest("manifest.json")     # after you have checked the manifest

ds["OP"].quality("nominal").describe()             # rank, duplicates, collinearity
hm.quicklook(ds, out="reports/")                   # a PDF of everything worth seeing

bank = hm.ModelBank.fit(ds, hm.Mahalanobis(threshold="empirical", alpha=1e-3))
hm.detection_report(bank, ds, out="reports/detection.pdf", at_alpha=1e-3)
```

`templates/01_quicklook_and_detection.ipynb` walks through this with commentary;
`templates/02_detectability.ipynb` covers the POD analysis. Both run on synthetic data
if you have none of your own.

## Why a library and not a script

On this very dataset, two sensors are literally the same column
(`HPOTP_w` ≡ `PBOBP_w`, r = 1.000000) and another pair is collinear to r = 0.9995.
The nominal covariance is therefore singular, and 16 of its 26 eigenvalues fall below
1e-3. Nothing crashes — which is the problem. Asking three chains for the same 0.1%
false alarm rate:

| chain | sensors | components | actual false alarm rate | detected |
|---|---|---|---|---|
| drop redundant + Ledoit-Wolf + Cholesky | 24/26 | 24 | 0.20 % | 100 % |
| keep everything + empirical covariance | 26/26 | 26 | **4.73 %** | 100 % |
| truncated eigendecomposition (99 % var.) | 26/26 | 7 | 0.13 % | 99.7 % |

The middle row is what a reasonable-looking script does, and it is wrong by a factor of
47 — the chi-square threshold assumes more degrees of freedom than the data carry.

So `hmslib.Mahalanobis` removes redundant sensors *before* estimating anything, never
forms an explicit inverse (Cholesky of `Σ + λI`, or a truncated eigendecomposition),
computes three comparable thresholds instead of one (chi-square, Hotelling `T²` with the
finite-sample correction, and an out-of-sample empirical quantile), and reports condition
number, effective rank and the ridge it actually needed in `det.diagnostics_`. Anything
questionable ends up in `det.warnings_`, in words.

`det.contributions(X)` decomposes the squared distance over sensors — the terms sum
exactly to `d²` — which turns detection into isolation.

## Detectability

When the fault intensity is swept inside the Monte Carlo loop, runs at the same
intensity still scatter, so "from which intensity is the fault visible" is answered by a
probability, not a number. `hmslib.analysis` fits the POD curve of each failure class
and reports `i50`, `i90`, and `i90_95` — the upper 95 % confidence bound on `i90`,
the conservative figure.

A POD curve only means something at a stated false alarm rate, so every result carries
the rate it holds at, and `at_alpha` re-calibrates the detector before the analysis.
An `i90` past the end of the sampled sweep is reported as an extrapolation rather than
as a result.

Validated against theory: for a Gaussian nominal cloud and a mean-shifting fault, `d²`
follows a non-central chi-square and the true POD is available in closed form. The
fitted `i90` lands within 3 % of it.

## Requirements

Python 3.9+, numpy, scipy, pandas, scikit-learn, matplotlib. Nothing else — no seaborn,
no yaml, no build step. `hmslib` is meant to be copied next to your scripts and
imported, because it was written for a machine with no internet access.

## Citation

```bibtex
@inproceedings{mattia2026rs25,
  author    = {Mattia, Francesco and Fabiani, Marco and Fiore, Matteo and Nasuti, Francesco},
  title     = {Data-Driven Health Monitoring and Fault Detection for the {RS-25}
               Staged Combustion Liquid Rocket Engine in {EcosimPro/ESPSS} Framework},
  booktitle = {Space Propulsion 2026},
  address   = {Bari, Italy},
  year      = {2026},
  note      = {SP2026\_170}
}
```

## License

To be defined.
