# rust_gp

Multi-band Gaussian-process light-curve fitter: a Matérn-3/2 GP whose exact
likelihood is evaluated by a Kalman filter in O(n), fitted by MAP with L-BFGS
and forward-mode-dual gradients. Returns per-band **peak magnitude, rise and
fade rates, and FWHM**, and the **blackbody temperature evolution, peak
temperature and cooling rate**, all with posterior uncertainties. CPU only, parallel across
sources with Rayon.
Structured to mirror [`villar-pso`](https://github.com/frenbox/villar-pso) and
[`sbpl-pso`](https://github.com/frenbox/sbpl-pso), and reads the same CSVs.

## Model

Zero-mean GP over flux normalised to peak 1, with a separable kernel across
time and wavelength (the avocado / PLAsTiCC kernel):

```
k((t, b), (t', b')) = σ_b σ_b' · exp(−(λ_b − λ_b')² / 2ℓ_λ²) · (1 + √3τ/ℓ) · exp(−√3τ/ℓ),   τ = |t − t'|
noise_i²            = flux_err_i² + jitter²
```

| Parameter | Meaning | Prior |
|-----------|---------|-------|
| `length_scale` ℓ | time scale (days) | soft box: median nightly cadence → max(baseline, 10 d) |
| `wavelength_scale` ℓ_λ | how strongly bands are tied (Å) | log-normal, median 6000 Å, σ_ln = 0.75 |
| `jitter` | extra white noise (flux) | log-normal, median 1 % of peak, σ_ln = 2 |
| `amp_<band>` σ_b | per-band GP amplitude (flux) | soft box: 1 %–1000 % of peak |

"Soft box" means flat inside, Gaussian walls of width 0.5 in ln outside.

As ℓ_λ → ∞ every band shares one curve shape and differs only in amplitude.
Smaller ℓ_λ lets colour evolve (e.g. a kilonova reddening), so the model
covers both cases without switching.

## Light-curve features

For every band with ≥ 3 detections, from the posterior mean curve (the
plotted line):

| Feature | Definition |
|---------|------------|
| `peak_time`, `peak_mag`, `peak_flux` | time and AB magnitude (ZP 23.9) of maximum flux |
| `t_rise` | days from the half-maximum crossing to the peak |
| `t_fade` | days from the peak to the half-maximum crossing after it |
| `fwhm` | `t_rise + t_fade` (days) |
| `rise_rate`, `fade_rate` | `2.5·log₁₀2 / t_rise`, `/ t_fade` (mag/day): the mean rate over the 0.753 mag between half-max and peak |
| `<name>_err` | robust 1σ, (p84 − p16)/2, over 100 joint posterior draws of the curve |
| `peak_bracketed`, `n_points` | peak lies strictly inside the band's data; detections in the band |

Rules that keep the numbers honest:

* **No extrapolation.** A band is searched only between its first and last
  data point. Beyond the data a zero-mean GP always decays to half-max, so a
  crossing there would be an artefact; an unobserved rise or fade is `None`.
* **Bracketed peaks only.** If the brightest point is the band's first or last
  data point, the true peak may lie outside the data and all features are
  `None` (`peak_bracketed = False`).
* **Non-detections pin the rise.** ZTF usually first detects a transient
  already above half-max. Upper limits (`diffmaglim`) before the first or
  after the last detection enter as `flux = 0 ± f_lim/5` — the 1σ noise of a
  difference-image measurement at that epoch — and extend the search range.
  Limits inside the detection window are dropped (a non-detection while the
  source is bright is usually a failed subtraction). Opt out with
  `upper_limits=False` / `GpConfig::use_upper_limits = false`.

Coverage on the 392 transient-like band light curves (baseline < 200 d, ≥ 3 points)
in `villar-pso/photometry`:

| | peak | rise | fade | FWHM |
|---|---|---|---|---|
| detections only | 82 % | 42 % | 60 % | 34 % |
| + non-detections (default) | 92 % | 86 % | 75 % | 70 % |

Uncertainties come from exact forward-filter/backward-sample draws (O(n) per
draw) and are conditional on the fitted hyperparameters. Tests check the draws
reproduce the smoother's moments and that features of a known two-band Bazin
transient are recovered within ~1σ, including a rise seen only through limits.

## Temperature and cooling rate

The GP is already two-dimensional — time × wavelength through the band kernel —
so its joint posterior gives every band's flux at every epoch. At each epoch
(spacing min(1 d, ℓ/10)) the SED is fitted with `A · B_ν(λ_b, T)`, weighted by
the posterior variances, with `A` profiled out.

| Output | Definition |
|--------|------------|
| `times`, `log_temp`, `log_temp_err` | per-epoch log₁₀ T (K) with 1σ errors |
| `bands` | bands in every SED (one set per source) |
| `ref_band`, `peak_time`, `log_temp_peak` | T at the light-curve peak of the best-sampled band with a bracketed peak |
| `cooling_rate` | OLS slope d log₁₀T/dt (dex/day) over `[t_peak, t_peak + 30 d]`; negative = cooling |
| `log_temp_latest` | T at the last usable epoch |

* **One band set per source.** SN SEDs are not blackbodies (line blanketing
  suppresses g), so g−r and r−i imply different T; letting bands drop in and
  out made T(t) jump by ~2 kK. Bands with ≥ 3 detections form the set, and an
  epoch is used only where all of them are inside their detection range at
  SNR ≥ 3 (falling back to the best pair if that leaves < 5 epochs).
* **Errors** repeat every blackbody fit on joint posterior draws, so they
  carry the cross-band and cross-epoch correlations of the GP.
* **Observed frame.** No redshift (T_rest = (1+z)·T) or extinction
  correction. With ZTF g/r/i (4770–7625 Å) T ≳ 15 000 K sits on the
  Rayleigh–Jeans tail and is weakly constrained; fits at the search bounds
  (log T ∉ (3.3, 5.0)) are dropped. With two bands the fit is exact (one
  colour), so only the posterior spread constrains T.

Same units and sign as boom-astro/lightcurve-fitting's `thermal_cooling_rate`,
which fits a linear log T(t) over all data with PSO; the 30-day window here
avoids mixing photospheric cooling with late nebular colour changes (on
ZTF20aavzffg the all-epoch slope is +0.0011 dex/d, the 30-day one −0.0034).
A synthetic g/r/i transient cooling at −0.0060 dex/d is recovered as
−0.0060 ± 0.0001, with log T_peak within 2σ.

```python
res = rust_gp.fit("ZTF24aacrfok.csv")
th = res["thermal"]
print(10**th["log_temp_peak"], th["cooling_rate"], th["cooling_rate_err"])
# 9781 K, -0.0172, 0.0022 dex/day
```

## Why this design

* **Exact O(n) likelihood.** A Matérn-3/2 process has an exact two-variable
  state `[f, f']`. Stacking one pair per band gives a `2B`-dimensional linear
  Gaussian state-space model whose Kalman log-likelihood *is* the dense-GP
  log-likelihood. It matches dense Cholesky to ~10⁻¹¹ at every size tested,
  handles irregular sampling and simultaneous epochs natively, and scales as
  O(n) instead of the O(n³) of a dense Cholesky factorisation.
* **MAP + L-BFGS, not PSO.** With a Matérn kernel and weak priors the
  posterior is close to unimodal (all starts agree on ~97 % of real sources,
  100 % of synthetic ones), so thousands of PSO evaluations buy nothing. Each
  fit runs 3 L-BFGS starts spread across the ℓ range and keeps the best.
* **Exact gradients for free.** The Kalman recursion is written once, generic
  over the scalar type; evaluating it with dual numbers gives the full gradient
  in the same pass (≈ 3–4× the cost of a plain evaluation).
* **Priors keep sparse fits sane.** They stop ℓ collapsing below the cadence
  (fitting noise) or running past the baseline — the failure mode of
  early-time light curves with a handful of points.
* **Warm starts for streaming.** When an alert adds a point, refit from the
  previous hyperparameters with one start (`fit_warm`): far fewer iterations,
  and it lands on the same optimum as a cold refit.
* **No GPU.** Each fit is a chain of tiny dependent matrix updates, which
  GPUs run poorly; throughput comes from fitting sources in parallel across
  CPU cores with Rayon.

## Build & install the Python extension

```bash
cd rust_gp
pip install maturin
maturin develop --release --features python
python -c "import rust_gp; print(dir(rust_gp))"   # ['fit', 'fit_many', 'predict', ...]
```

## Python API

### `rust_gp.fit(csv_path, **kwargs) → dict`

| Key | Type | Description |
|-----|------|-------------|
| `thermal` | dict | `{times, log_temp, log_temp_err, chi2, bands, ref_band, peak_time, log_temp_peak(_err), cooling_rate(_err), log_temp_latest(_err)}` — see [Temperature and cooling rate](#temperature-and-cooling-rate) |
| `features` | dict | `{band: {peak_time, peak_mag, peak_flux, t_rise, t_fade, fwhm, rise_rate, fade_rate, <name>_err, peak_bracketed, n_points}}` — see [Light-curve features](#light-curve-features) |
| `params` | dict | `length_scale` (d), `wavelength_scale` (Å), `jitter`, `amp_<band>` (flux), each with `<name>_ln_err`: Laplace 1σ error on ln(param) ≈ fractional error (`None` if the Hessian is not PD) |
| `obs` | list of dicts | `{time, flux, flux_err, band, upper_limit}` in physical flux (µJy, ZP 23.9); limits are `flux 0 ± f_lim/5` |
| `log_likelihood`, `log_posterior` | float | at the optimum |
| `n_obs`, `n_limits`, `n_bands` | int | detections / non-detections / bands used |
| `n_starts`, `n_starts_agree` | int | starts run / starts within 10⁻³ nats of the best |
| `converged` | bool | best start met a convergence criterion |

Keywords: `n_starts=3`, `max_iters=200`, `min_obs=5`,
`length_scale_bounds=None` (tuple of days), `wavelength_scale_prior=(6000.0, 0.75)`,
`jitter_prior=(0.01, 2.0)`, `laplace_errors=True`, `upper_limits=True`,
`features=True`, `thermal=True`, `n_samples=100` (posterior draws for feature and
temperature errors; 0 = values only, much faster), `seed=0`.

### `rust_gp.fit_many(csv_paths, n_threads=None, **kwargs) → list[dict]`

Fits in parallel across cores (the GIL is released). One dict per path in
input order; a failed path yields `{"error": "..."}` in its slot.

### `rust_gp.features(result, n_samples=100, seed=0, min_band_points=3) → dict`

Recompute the features dict from a `fit()` result (e.g. with more draws).

```python
res = rust_gp.fit("ZTF20aavzffg.csv")
r = res["features"]["r"]
print(r["peak_mag"], r["rise_rate"], r["fade_rate"], r["fwhm"], r["fwhm_err"])
# 15.10  0.072  0.048  26.2  0.5
```

### `rust_gp.thermal(result, n_samples=100, seed=0, min_snr=3.0, min_band_points=3, cooling_window=30.0) → dict`

Recompute the temperature evolution from a `fit()` result (e.g. another
cooling window; `None` = all post-peak epochs).

### `rust_gp.predict(result, t_dense, band, include_noise=False) → (mean, std)`

Posterior mean and standard deviation of the latent light curve in `band`.
Takes the whole `fit()` result: a GP prediction conditions on the data, so it
needs `obs` as well as `params`. `include_noise=True` adds the jitter.

```python
import rust_gp, numpy as np

res = rust_gp.fit("ZTF23absdibx.csv")
t = np.linspace(2460280, 2460380, 1000)
mean_r, std_r = rust_gp.predict(res, t.tolist(), "r")
```

## Rust API

```toml
[dependencies]
rust_gp = { git = "https://github.com/frenbox/rust_gp.git" }
```

```rust
use rust_gp::{fit, fit_warm, load_csv, predict, prepare, GpConfig};

let config = GpConfig::default();
let obs = load_csv("ZTF23absdibx.csv")?;
let result = fit(&obs, &config)?;
println!("ℓ = {:.2} d", result.hyper.length_scale);

// Streaming: a new alert arrives — refit from the previous optimum.
let updated = fit_warm(&obs_with_new_point, &config, &result.hyper)?;

// Posterior on a grid.
let prep = prepare(&obs_with_new_point, &config)?;
let curve = predict(&prep, &updated.hyper, &t_grid, "r", false)?;

// Peak, rise/fade rates, FWHM per band; temperature and cooling rate.
let feats = rust_gp::features::light_curve_features(&prep, &updated.hyper, &Default::default());
let thermal = rust_gp::thermal::thermal_evolution(&prep, &updated.hyper, &Default::default());
```

Use `load_csv_with_limits` (or `load_csv_for(path, &config)`) to include
non-detections; `load_csv` reads detections only.

`fit_many_csv(&paths, &config)` fits on the current Rayon pool. Lower-level
pieces are public too: `kalman::log_likelihood` (generic over `f64` /
`dual::Dual<N>`), `neg_log_posterior_grad`, `optim::minimize`, and
`dense::{log_likelihood, predict}` as an O(n³) reference.

## Plotting script

```bash
pip install numpy matplotlib
python scripts/fit_gp.py ../lc_fitting_comparison/data/photometry/ZTF23absdibx.csv
python scripts/fit_gp.py ../lc_fitting_comparison/data/photometry/ --max-files 20 --flux
```

Writes `<repo>/gp_results/features.csv` (one row per source × band, every
feature with its error) and `<name>_gp.png`: data, non-detections (▽), GP
mean ± 1σ, the peak (★) and the FWHM bar at half maximum, in magnitudes
(default) or flux. `--no-limits` ignores non-detections, `--no-plots` writes
only the table.

## Tests & benchmarks

```bash
# Correctness (no data needed, < 1 s): Kalman vs dense likelihood and
# posterior, dual vs finite-difference gradients, hyperparameter recovery,
# start agreement, warm vs cold refit, edge cases
cargo test --release --test test_gp -- --nocapture

# Features: posterior draws vs smoother moments, exact on a Gaussian, Bazin
# recovery, unobserved rise/fade → None, rise recovered from upper limits
cargo test --release --test test_features -- --nocapture

# Thermal: exact blackbody recovery, peak T and cooling rate of a synthetic
# cooling transient, single band → no temperature
cargo test --release --test test_thermal -- --nocapture

# Benchmarks — run on the deployment machine; timings are hardware-specific.
# Single-core timing table on synthetic data (dense vs Kalman, fit, warm refit)
cargo run --release --bin likelihood-bench

# Real data: throughput by thread count, per-length breakdown, start agreement
cargo run --release --bin fit-bench -- path/to/photometry_dir --threads 1,8,32
```

## Known limitations

* **Zero mean.** Far from data the GP falls back to zero flux. That is the
  right prior for transients (they fade to nothing) but makes persistent
  variables and AGN dip toward zero inside seasonal gaps. A per-band constant
  mean can be marginalised exactly by adding one constant state per band
  (still O(n), ~2× cost); not implemented yet.
* **No outlier handling.** One bad point inflates the jitter. A robust
  likelihood would break the exact Kalman form; an iterative clip on smoothed
  residuals is the likely route.
* **Upper limits are approximate.** Entering a non-detection as `0 ± f_lim/5`
  ignores that the true flux was merely *below* the limit; a source just
  under the limit pulls the curve slightly low, so rises from limits come out
  a little fast (tests: ~0.5 d short of a 7–8 d rise, within 1σ). `isdiffpos`
  is ignored, as in the sibling repos.
* **Features are conditional on the hyperparameters** — the MAP length scale
  is not marginalised over, so errors are somewhat optimistic for sparse
  light curves.
* **Multi-start disagreement** occurs on ~3 % of sources, where short- and
  long-ℓ explanations compete; the best start is always the one reported, and
  `n_starts_agree` exposes the cases.

## Project structure

```
rust_gp/
├── Cargo.toml
├── src/
│   ├── lib.rs               # CSV loading, preparation, priors, fit driver, PyO3
│   ├── kalman.rs            # state-space Matérn-3/2: likelihood, RTS smoother, posterior draws
│   ├── features.rs          # peak, rise/fade rate, FWHM per band (+ errors)
│   ├── thermal.rs           # blackbody T(t), T at peak, cooling rate (+ errors)
│   ├── dual.rs              # forward-mode dual numbers (Scalar trait)
│   ├── optim.rs             # L-BFGS with backtracking line search
│   ├── dense.rs             # O(n³) reference GP (tests / benchmarks)
│   ├── linalg.rs            # small dense Cholesky helpers
│   ├── rng.rs               # SplitMix64 RNG for posterior sampling
│   ├── synthetic.rs         # exact GP-prior draws for tests / benchmarks
│   └── bin/
│       ├── fit_bench.rs
│       └── likelihood_bench.rs
├── scripts/fit_gp.py        # fit + plot (paths anchored to the repo root)
└── tests/
    ├── test_gp.rs
    ├── test_features.rs
    └── test_thermal.rs
```
