//! Multi-band Gaussian-process light-curve fitter.
//!
//! Model: a zero-mean GP over normalised flux with the separable kernel
//!
//!   k((t, b), (t', b')) = σ_b σ_b' · exp(−(λ_b − λ_b')² / 2ℓ_λ²) · k₃/₂(|t − t'|; ℓ)
//!
//! plus per-point noise `flux_err² + jitter²`. Large `ℓ_λ` recovers a single
//! curve shape shared by every band (per-band amplitude only); smaller `ℓ_λ`
//! lets colour evolve. See [`kalman`] for the state-space form.
//!
//! Fitting: maximise `ln p(y | θ) + ln p(θ)` (MAP) over
//! `θ = [ln ℓ, ln ℓ_λ, ln jitter, ln σ_b …]` with L-BFGS from a few starts.
//! The likelihood is an exact O(n) Kalman filter and its gradient comes from
//! forward-mode dual numbers in the same pass. Weak priors keep `ℓ` between
//! the cadence and the baseline, which is what stabilises sparse early-time
//! light curves.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub mod dense;
pub mod dual;
pub mod features;
pub mod kalman;
pub(crate) mod linalg;
pub mod optim;
pub mod rng;
pub mod synthetic;
pub mod thermal;

use dual::{Dual, Scalar};
use optim::{LbfgsOptions, LbfgsResult};

/// Position of `ln ℓ` (time length scale, days) in `θ`.
pub const IDX_LS: usize = 0;
/// Position of `ln ℓ_λ` (wavelength length scale, 1000 Å) in `θ`.
pub const IDX_WL: usize = 1;
/// Position of `ln jitter` (normalised flux) in `θ`.
pub const IDX_JIT: usize = 2;
/// Position of the first `ln σ_b` (normalised flux) in `θ`; one per band.
pub const IDX_AMP: usize = 3;
/// Maximum number of distinct bands (u, g, r, i, z, y).
pub const MAX_BANDS: usize = 6;

// ---------------------------------------------------------------------------
// Bands, magnitudes, CSV loading (same conventions as villar-pso / sbpl-pso)
// ---------------------------------------------------------------------------

/// Effective central wavelength (Å) for supported filter names.
pub fn band_wavelength(band: &str) -> Option<f64> {
    match band {
        "u" | "lsstu" => Some(3540.0),
        "g" | "ztfg" | "lsstg" | "ZTF_g" => Some(4770.0),
        "r" | "ztfr" | "lsstr" | "ZTF_r" => Some(6231.0),
        "i" | "ztfi" | "lssti" | "ZTF_i" => Some(7625.0),
        "z" | "ztfz" | "lsstz" => Some(9100.0),
        "y" | "lssty" => Some(9710.0),
        _ => None,
    }
}

/// AB magnitude → flux (zeropoint 23.9, i.e. µJy).
pub fn mag_to_flux(mag: f64, mag_err: f64) -> (f64, f64) {
    let flux = 10.0_f64.powf((23.9 - mag) / 2.5);
    let flux_err = flux * mag_err * std::f64::consts::LN_10 / 2.5;
    (flux, flux_err)
}

/// One photometric observation in flux space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Obs {
    /// JD or MJD (days), whatever the input used.
    pub time: f64,
    pub flux: f64,
    pub flux_err: f64,
    /// Short band name: "u", "g", "r", "i", "z" or "y".
    pub band: String,
    /// A non-detection entered as `flux = 0 ± f_lim/5` (see [`load_csv_with_limits`]).
    #[serde(default)]
    pub upper_limit: bool,
}

/// Load a ZTF/LSST-style photometry CSV and convert magnitudes to flux.
///
/// Accepts `jd` (or `mjd`), `magpsf` (or `mag`), `sigmapsf` (or `mag_err`), and
/// either `fid` (1=g, 2=r, 3=i) or a `filter` string column. Rows with missing
/// or non-finite values (e.g. ZTF non-detections) are skipped.
pub fn load_csv(path: &str) -> Result<Vec<Obs>, String> {
    read_csv(path, false)
}

/// [`load_csv`] plus non-detections as Gaussian points at zero flux.
///
/// A row without a magnitude but with a 5σ limiting magnitude (`diffmaglim`
/// or `limmag`) becomes `flux = 0`, `flux_err = f_lim / 5` — the noise level
/// of a difference-image measurement at that epoch. Only limits before the
/// first or after the last detection (any band) are kept: inside that window
/// the detections already constrain the curve, and a non-detection while the
/// source is bright is usually a failed subtraction.
pub fn load_csv_with_limits(path: &str) -> Result<Vec<Obs>, String> {
    read_csv(path, true)
}

fn read_csv(path: &str, with_limits: bool) -> Result<Vec<Obs>, String> {
    let mut rdr = csv::ReaderBuilder::new()
        .flexible(true)
        .from_path(path)
        .map_err(|e| format!("Cannot open {path}: {e}"))?;
    let headers = rdr.headers().map_err(|e| format!("Cannot read headers: {e}"))?.clone();
    let col = |name: &str| headers.iter().position(|h| h == name);

    let time_col = col("jd").or_else(|| col("mjd")).ok_or("CSV must have 'jd' or 'mjd' column")?;
    let mag_col = col("magpsf").or_else(|| col("mag")).ok_or("CSV must have 'magpsf' or 'mag' column")?;
    let err_col = col("sigmapsf")
        .or_else(|| col("mag_err"))
        .ok_or("CSV must have 'sigmapsf' or 'mag_err' column")?;
    let lim_col = if with_limits { col("diffmaglim").or_else(|| col("limmag")) } else { None };
    let fid_col = col("fid");
    let filter_col = col("filter");

    let mut obs = Vec::new();
    let mut limits = Vec::new();
    for record in rdr.records() {
        let rec = record.map_err(|e| format!("Parse error: {e}"))?;
        let parse = |c: usize| rec.get(c).and_then(|s| s.trim().parse::<f64>().ok());
        let Some(time) = parse(time_col) else { continue };
        let band = if let Some(fc) = fid_col {
            match rec.get(fc).and_then(|s| s.trim().parse::<u32>().ok()) {
                Some(1) => "g",
                Some(2) => "r",
                Some(3) => "i",
                _ => continue,
            }
        } else if let Some(fc) = filter_col {
            match rec.get(fc).map(|s| s.trim()) {
                Some("u" | "lsstu") => "u",
                Some("g" | "ZTF_g" | "ztfg" | "lsstg") => "g",
                Some("r" | "ZTF_r" | "ztfr" | "lsstr") => "r",
                Some("i" | "ZTF_i" | "ztfi" | "lssti") => "i",
                Some("z" | "ztfz" | "lsstz") => "z",
                Some("y" | "lssty") => "y",
                _ => continue,
            }
        } else {
            continue;
        };
        match (parse(mag_col), parse(err_col)) {
            (Some(mag), Some(mag_err)) if mag_err > 0.0 => {
                let (flux, flux_err) = mag_to_flux(mag, mag_err);
                if flux.is_finite() && flux_err.is_finite() && flux > 0.0 {
                    obs.push(Obs { time, flux, flux_err, band: band.to_string(), upper_limit: false });
                }
            }
            (None, _) => {
                if let Some(lim) = lim_col.and_then(parse) {
                    let (f_lim, _) = mag_to_flux(lim, 0.0);
                    if f_lim.is_finite() && f_lim > 0.0 {
                        limits.push(Obs { time, flux: 0.0, flux_err: f_lim / 5.0, band: band.to_string(), upper_limit: true });
                    }
                }
            }
            _ => {}
        }
    }
    if obs.is_empty() {
        return Err(format!("No valid u/g/r/i/z/y observations in {path}"));
    }
    let first = obs.iter().map(|o| o.time).fold(f64::INFINITY, f64::min);
    let last = obs.iter().map(|o| o.time).fold(f64::NEG_INFINITY, f64::max);
    obs.extend(limits.into_iter().filter(|o| o.time < first || o.time > last));
    Ok(obs)
}

/// All `*.csv` files in `dir`, sorted.
pub fn find_csv_files(dir: &str) -> Vec<String> {
    let mut paths: Vec<String> = std::fs::read_dir(dir)
        .expect("Cannot read directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "csv"))
        .map(|e| e.path().to_string_lossy().to_string())
        .collect();
    paths.sort();
    paths
}

// ---------------------------------------------------------------------------
// Configuration and priors
// ---------------------------------------------------------------------------

/// Fit settings and prior hyper-hyperparameters.
#[derive(Debug, Clone)]
pub struct GpConfig {
    /// Number of L-BFGS starts, spread geometrically across the `ℓ` range.
    pub n_starts: usize,
    /// L-BFGS iteration cap per start.
    pub max_iters: usize,
    /// Convergence tolerance on `max |∂(−ln posterior)/∂θ|`.
    pub grad_tol: f64,
    /// Sources with fewer usable points are rejected.
    pub min_obs: usize,
    /// Soft bounds on `ℓ` (days). `None`: from the median nightly cadence up
    /// to the baseline.
    pub length_scale_bounds: Option<(f64, f64)>,
    /// Log-normal prior on `ℓ_λ`: (median in Å, σ of ln). avocado fixes 6000 Å.
    pub wavelength_scale_prior: (f64, f64),
    /// Log-normal prior on the jitter: (median as a fraction of peak flux, σ of ln).
    pub jitter_prior: (f64, f64),
    /// Compute Laplace (inverse-Hessian) errors on `ln θ`.
    pub laplace_errors: bool,
    /// When loading CSVs, include non-detections outside the detection
    /// window (see [`load_csv_with_limits`]).
    pub use_upper_limits: bool,
}

impl Default for GpConfig {
    fn default() -> Self {
        GpConfig {
            n_starts: 3,
            max_iters: 200,
            grad_tol: 1e-6,
            min_obs: 5,
            length_scale_bounds: None,
            wavelength_scale_prior: (6000.0, 0.75),
            jitter_prior: (0.01, 2.0),
            laplace_errors: true,
            use_upper_limits: true,
        }
    }
}

/// Width (in ln units) of the quadratic walls outside a soft box.
const BOX_WIDTH: f64 = 0.5;
/// Smallest default upper edge of the `ℓ` soft box (days).
const MIN_LS_UPPER: f64 = 10.0;
/// Soft box for the per-band amplitudes in normalised flux.
const AMP_BOUNDS: (f64, f64) = (0.01, 10.0);

/// Priors in `θ` space, resolved for one source.
#[derive(Debug, Clone)]
pub struct Prior {
    pub ln_ls_lo: f64,
    pub ln_ls_hi: f64,
    pub ln_wl_mu: f64,
    pub wl_sigma: f64,
    pub ln_jit_mu: f64,
    pub jit_sigma: f64,
}

#[inline]
fn soft_box<T: Scalar>(x: T, lo: f64, hi: f64) -> T {
    let v = x.val();
    if v < lo {
        let z = (T::cst(lo) - x).scale(1.0 / BOX_WIDTH);
        -(z * z).scale(0.5)
    } else if v > hi {
        let z = (x - T::cst(hi)).scale(1.0 / BOX_WIDTH);
        -(z * z).scale(0.5)
    } else {
        T::cst(0.0)
    }
}

#[inline]
fn normal<T: Scalar>(x: T, mu: f64, sigma: f64) -> T {
    let z = (x - T::cst(mu)).scale(1.0 / sigma);
    -(z * z).scale(0.5)
}

impl Prior {
    /// `ln p(θ)` up to a constant.
    pub fn log_prob<T: Scalar>(&self, theta: &[T]) -> T {
        let (amp_lo, amp_hi) = (AMP_BOUNDS.0.ln(), AMP_BOUNDS.1.ln());
        let mut lp = soft_box(theta[IDX_LS], self.ln_ls_lo, self.ln_ls_hi)
            + normal(theta[IDX_WL], self.ln_wl_mu, self.wl_sigma)
            + normal(theta[IDX_JIT], self.ln_jit_mu, self.jit_sigma);
        for &t in &theta[IDX_AMP..] {
            lp += soft_box(t, amp_lo, amp_hi);
        }
        lp
    }
}

// ---------------------------------------------------------------------------
// Preparation
// ---------------------------------------------------------------------------

/// One source in fitting form: time-sorted, shifted, flux-normalised.
#[derive(Debug, Clone)]
pub struct PreparedSource {
    /// Times in days since [`Self::t_ref`], ascending.
    pub times: Vec<f64>,
    /// Flux divided by [`Self::flux_scale`].
    pub flux: Vec<f64>,
    pub flux_err: Vec<f64>,
    /// Index into [`Self::bands`] per point.
    pub band_idx: Vec<usize>,
    /// Whether each point is a non-detection (`flux = 0 ± f_lim/5`).
    pub upper_limit: Vec<bool>,
    /// Bands present, sorted by wavelength.
    pub bands: Vec<String>,
    /// Band wavelengths in units of 1000 Å (the unit of `ℓ_λ` inside `θ`).
    pub wavelengths: Vec<f64>,
    /// Earliest observation time (input units).
    pub t_ref: f64,
    /// Max |flux|, so normalised fluxes peak at 1.
    pub flux_scale: f64,
    pub prior: Prior,
}

impl PreparedSource {
    /// All points, detections and non-detections.
    pub fn n_obs(&self) -> usize {
        self.times.len()
    }

    /// Detections only.
    pub fn n_detections(&self) -> usize {
        self.upper_limit.iter().filter(|&&l| !l).count()
    }

    pub fn n_bands(&self) -> usize {
        self.bands.len()
    }

    /// Number of hyperparameters, `3 + n_bands`.
    pub fn n_params(&self) -> usize {
        IDX_AMP + self.bands.len()
    }

    /// Names of the entries of `θ`, without the `ln`.
    pub fn param_names(&self) -> Vec<String> {
        let mut names = vec!["length_scale".to_string(), "wavelength_scale".into(), "jitter".into()];
        names.extend(self.bands.iter().map(|b| format!("amp_{b}")));
        names
    }

    /// `θ` (normalised, log space) → physical hyperparameters.
    pub fn hyper_from_theta(&self, theta: &[f64]) -> GpHyper {
        GpHyper {
            length_scale: theta[IDX_LS].exp(),
            wavelength_scale: theta[IDX_WL].exp() * 1000.0,
            jitter: theta[IDX_JIT].exp() * self.flux_scale,
            amplitudes: self
                .bands
                .iter()
                .enumerate()
                .map(|(i, b)| (b.clone(), theta[IDX_AMP + i].exp() * self.flux_scale))
                .collect(),
        }
    }

    /// Physical hyperparameters → `θ`. Bands absent from `hyper` (e.g. a band
    /// that first appeared in the latest alert) get the default start.
    pub fn theta_from_hyper(&self, hyper: &GpHyper) -> Vec<f64> {
        let mut theta = self.default_theta(hyper.length_scale.max(1e-6).ln());
        theta[IDX_WL] = (hyper.wavelength_scale / 1000.0).max(1e-6).ln();
        theta[IDX_JIT] = (hyper.jitter / self.flux_scale).max(1e-8).ln();
        for (i, b) in self.bands.iter().enumerate() {
            if let Some((_, a)) = hyper.amplitudes.iter().find(|(name, _)| name == b) {
                theta[IDX_AMP + i] = (a / self.flux_scale).max(1e-6).ln();
            }
        }
        theta
    }

    /// A starting `θ` with the given `ln ℓ`: prior medians for `ℓ_λ` and the
    /// jitter, per-band RMS of the normalised flux for the amplitudes.
    fn default_theta(&self, ln_ls: f64) -> Vec<f64> {
        let mut theta = vec![0.0; self.n_params()];
        theta[IDX_LS] = ln_ls;
        theta[IDX_WL] = self.prior.ln_wl_mu;
        theta[IDX_JIT] = self.prior.ln_jit_mu;
        let mut sum2 = vec![0.0; self.n_bands()];
        let mut cnt = vec![0usize; self.n_bands()];
        for (y, &b) in self.flux.iter().zip(&self.band_idx) {
            sum2[b] += y * y;
            cnt[b] += 1;
        }
        for b in 0..self.n_bands() {
            let rms = (sum2[b] / cnt[b].max(1) as f64).sqrt();
            theta[IDX_AMP + b] = rms.clamp(0.05, 5.0).ln();
        }
        theta
    }
}

/// Validate, sort, shift and normalise one source's observations.
pub fn prepare(obs: &[Obs], config: &GpConfig) -> Result<PreparedSource, String> {
    let mut rows: Vec<(f64, f64, f64, f64, bool)> = Vec::with_capacity(obs.len()); // (t, λ, y, σ, limit)
    for o in obs {
        let Some(wl) = band_wavelength(&o.band) else { continue };
        if o.time.is_finite() && o.flux.is_finite() && o.flux_err.is_finite() && o.flux_err > 0.0 {
            rows.push((o.time, wl, o.flux, o.flux_err, o.upper_limit));
        }
    }
    let n_det = rows.iter().filter(|r| !r.4).count();
    if n_det < config.min_obs {
        return Err(format!("too little data: {n_det} usable detections (need {})", config.min_obs));
    }
    rows.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));

    // Bands present, ordered by wavelength. Names are canonicalised through
    // the wavelength so "ztfg" and "g" are one band.
    let mut wls: Vec<f64> = rows.iter().filter(|r| !r.4).map(|r| r.1).collect();
    wls.sort_by(f64::total_cmp);
    wls.dedup();
    let canonical = |wl: f64| -> String {
        ["u", "g", "r", "i", "z", "y"]
            .into_iter()
            .find(|b| band_wavelength(b) == Some(wl))
            .unwrap()
            .to_string()
    };
    let bands: Vec<String> = wls.iter().map(|&w| canonical(w)).collect();
    // Limits in a band with no detections carry no shape information.
    rows.retain(|r| wls.contains(&r.1));

    let t_ref = rows[0].0;
    let flux_scale = rows.iter().fold(0.0f64, |m, r| m.max(r.2.abs()));
    if !(flux_scale > 0.0) {
        return Err("all fluxes are zero".into());
    }
    let times: Vec<f64> = rows.iter().map(|r| r.0 - t_ref).collect();
    let band_idx = rows.iter().map(|r| wls.iter().position(|&w| w == r.1).unwrap()).collect();
    let flux = rows.iter().map(|r| r.2 / flux_scale).collect();
    let flux_err = rows.iter().map(|r| r.3 / flux_scale).collect();
    let upper_limit: Vec<bool> = rows.iter().map(|r| r.4).collect();

    // ℓ soft box: median gap between distinct nights → baseline. The upper
    // edge never drops below MIN_LS_UPPER: a first night of detections says
    // nothing about ℓ beyond its own span, and a sub-day cap would force the
    // curve to collapse immediately after the data. Detections only: years
    // of pre-discovery limits say nothing about the transient's time scale.
    let det_times: Vec<f64> = times.iter().zip(&upper_limit).filter(|(_, &l)| !l).map(|(&t, _)| t).collect();
    let baseline = det_times.last().unwrap() - det_times[0];
    let (ls_lo, ls_hi) = config.length_scale_bounds.unwrap_or_else(|| {
        let mut gaps: Vec<f64> = det_times.windows(2).map(|w| w[1] - w[0]).filter(|&g| g > 0.3).collect();
        gaps.sort_by(f64::total_cmp);
        let cadence = gaps.get(gaps.len() / 2).copied().unwrap_or(0.1).max(0.1);
        (cadence, baseline.max(4.0 * cadence).max(MIN_LS_UPPER))
    });

    let prior = Prior {
        ln_ls_lo: ls_lo.ln(),
        ln_ls_hi: ls_hi.ln(),
        ln_wl_mu: (config.wavelength_scale_prior.0 / 1000.0).ln(),
        wl_sigma: config.wavelength_scale_prior.1,
        ln_jit_mu: config.jitter_prior.0.ln(),
        jit_sigma: config.jitter_prior.1,
    };

    Ok(PreparedSource {
        times,
        flux,
        flux_err,
        band_idx,
        upper_limit,
        bands,
        wavelengths: wls.iter().map(|w| w / 1000.0).collect(),
        t_ref,
        flux_scale,
        prior,
    })
}

// ---------------------------------------------------------------------------
// Objective
// ---------------------------------------------------------------------------

/// `ln p(y | θ) + ln p(θ)`.
pub fn log_posterior<T: Scalar>(prep: &PreparedSource, theta: &[T]) -> T {
    kalman::log_likelihood(prep, theta) + prep.prior.log_prob(theta)
}

fn neg_log_post_dual<const N: usize>(prep: &PreparedSource, theta: &[f64], grad: &mut [f64]) -> f64 {
    let th: Vec<Dual<N>> = theta.iter().enumerate().map(|(i, &v)| Dual::var(v, i)).collect();
    let lp = log_posterior(prep, &th);
    for i in 0..N {
        grad[i] = -lp.d[i];
    }
    -lp.v
}

/// `−ln posterior` and its exact gradient (forward-mode duals, one pass).
pub fn neg_log_posterior_grad(prep: &PreparedSource, theta: &[f64], grad: &mut [f64]) -> f64 {
    macro_rules! dispatch {
        ($($n:literal)*) => {
            match theta.len() {
                $($n => neg_log_post_dual::<$n>(prep, theta, grad),)*
                n => panic!("unsupported number of hyperparameters: {n}"),
            }
        };
    }
    dispatch!(4 5 6 7 8 9)
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

/// GP hyperparameters in physical units.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpHyper {
    /// Matérn-3/2 time length scale ℓ (days).
    pub length_scale: f64,
    /// Wavelength correlation length ℓ_λ (Å).
    pub wavelength_scale: f64,
    /// Extra white noise added to every point (flux units).
    pub jitter: f64,
    /// Per-band GP amplitude σ_b (flux units), by band name.
    pub amplitudes: Vec<(String, f64)>,
}

/// Outcome of one fit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpResult {
    pub hyper: GpHyper,
    /// Names matching [`Self::ln_err`]: `length_scale`, `wavelength_scale`,
    /// `jitter`, `amp_<band>`….
    pub param_names: Vec<String>,
    /// Laplace 1σ errors on the natural log of each hyperparameter (≈
    /// fractional errors). `None` if disabled or the Hessian is not PD.
    pub ln_err: Option<Vec<f64>>,
    pub log_likelihood: f64,
    pub log_posterior: f64,
    /// Detections used.
    pub n_obs: usize,
    /// Non-detections used.
    pub n_limits: usize,
    pub n_bands: usize,
    /// Starts that ended within 10⁻³ nats of the best — a direct check that
    /// the posterior is unimodal for this source.
    pub n_starts_agree: usize,
    pub n_starts: usize,
    /// L-BFGS iterations / objective evaluations, summed over starts.
    pub n_iters: usize,
    pub n_evals: usize,
    /// Whether the best start met a convergence criterion.
    pub converged: bool,
}

/// Posterior of the latent light curve at query times (flux units).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Fitting
// ---------------------------------------------------------------------------

fn lbfgs_options(config: &GpConfig) -> LbfgsOptions {
    LbfgsOptions { max_iters: config.max_iters, grad_tol: config.grad_tol, ..Default::default() }
}

/// Starting points: `ℓ` spread geometrically inside its soft box.
fn starts(prep: &PreparedSource, n: usize) -> Vec<Vec<f64>> {
    let (lo, hi) = (prep.prior.ln_ls_lo, prep.prior.ln_ls_hi);
    (0..n.max(1))
        .map(|j| {
            let u = (j as f64 + 0.5) / n.max(1) as f64;
            prep.default_theta(lo + u * (hi - lo))
        })
        .collect()
}

fn run(prep: &PreparedSource, config: &GpConfig, starts: Vec<Vec<f64>>) -> GpResult {
    let opts = lbfgs_options(config);
    let runs: Vec<LbfgsResult> = starts
        .into_iter()
        .map(|x0| optim::minimize(|x, g| neg_log_posterior_grad(prep, x, g), x0, &opts))
        .collect();
    let best = runs
        .iter()
        .filter(|r| r.f.is_finite())
        .min_by(|a, b| a.f.total_cmp(&b.f))
        .unwrap_or(&runs[0]);
    let n_starts_agree = runs.iter().filter(|r| (r.f - best.f).abs() < 1e-3).count();

    let ln_err = if config.laplace_errors { laplace_ln_errors(prep, &best.x) } else { None };
    GpResult {
        hyper: prep.hyper_from_theta(&best.x),
        param_names: prep.param_names(),
        ln_err,
        log_likelihood: kalman::log_likelihood(prep, &best.x),
        log_posterior: -best.f,
        n_obs: prep.n_detections(),
        n_limits: prep.n_obs() - prep.n_detections(),
        n_bands: prep.n_bands(),
        n_starts_agree,
        n_starts: runs.len(),
        n_iters: runs.iter().map(|r| r.n_iters).sum(),
        n_evals: runs.iter().map(|r| r.n_evals).sum(),
        converged: best.converged,
    }
}

/// `sqrt(diag(H⁻¹))` with `H` the Hessian of `−ln posterior` at `theta`,
/// by central differences of the exact gradient.
fn laplace_ln_errors(prep: &PreparedSource, theta: &[f64]) -> Option<Vec<f64>> {
    let n = theta.len();
    let h = 1e-4;
    let mut hess = vec![0.0; n * n];
    let (mut gp, mut gm) = (vec![0.0; n], vec![0.0; n]);
    let mut x = theta.to_vec();
    for i in 0..n {
        x[i] = theta[i] + h;
        neg_log_posterior_grad(prep, &x, &mut gp);
        x[i] = theta[i] - h;
        neg_log_posterior_grad(prep, &x, &mut gm);
        x[i] = theta[i];
        for j in 0..n {
            hess[i * n + j] = (gp[j] - gm[j]) / (2.0 * h);
        }
    }
    for i in 0..n {
        for j in 0..i {
            let s = 0.5 * (hess[i * n + j] + hess[j * n + i]);
            hess[i * n + j] = s;
            hess[j * n + i] = s;
        }
    }
    if !linalg::cholesky(&mut hess, n) {
        return None;
    }
    let mut errs = Vec::with_capacity(n);
    let mut e = vec![0.0; n];
    for i in 0..n {
        e.iter_mut().for_each(|v| *v = 0.0);
        e[i] = 1.0;
        linalg::chol_solve(&hess, n, &mut e);
        errs.push(e[i].max(0.0).sqrt());
    }
    errs.iter().all(|v| v.is_finite()).then_some(errs)
}

/// Fit an already-prepared source from `config.n_starts` cold starts.
pub fn fit_prepared(prep: &PreparedSource, config: &GpConfig) -> GpResult {
    run(prep, config, starts(prep, config.n_starts))
}

/// Fit from a single start at `previous` — the update path when a new alert
/// adds a point to a light curve that was already fitted.
pub fn fit_prepared_warm(prep: &PreparedSource, config: &GpConfig, previous: &GpHyper) -> GpResult {
    run(prep, config, vec![prep.theta_from_hyper(previous)])
}

/// Prepare and fit (cold).
pub fn fit(obs: &[Obs], config: &GpConfig) -> Result<GpResult, String> {
    Ok(fit_prepared(&prepare(obs, config)?, config))
}

/// Prepare and fit (warm start from a previous fit of the same source).
pub fn fit_warm(obs: &[Obs], config: &GpConfig, previous: &GpHyper) -> Result<GpResult, String> {
    Ok(fit_prepared_warm(&prepare(obs, config)?, config, previous))
}

/// Load a CSV and fit it.
pub fn fit_csv(path: &str, config: &GpConfig) -> Result<GpResult, String> {
    fit(&load_csv_for(path, config)?, config)
}

/// [`load_csv_with_limits`] or [`load_csv`], per `config.use_upper_limits`.
pub fn load_csv_for(path: &str, config: &GpConfig) -> Result<Vec<Obs>, String> {
    if config.use_upper_limits { load_csv_with_limits(path) } else { load_csv(path) }
}

/// Fit many CSVs in parallel on the current Rayon pool, in input order.
pub fn fit_many_csv(paths: &[String], config: &GpConfig) -> Vec<Result<GpResult, String>> {
    paths.par_iter().map(|p| fit_csv(p, config)).collect()
}

/// Posterior mean and standard deviation of the light curve in `band` at
/// times `t` (same units as the input times). With `include_noise`, the
/// jitter is added to the variance (predictive scatter of a new point,
/// excluding its own reported error).
pub fn predict(
    prep: &PreparedSource,
    hyper: &GpHyper,
    t: &[f64],
    band: &str,
    include_noise: bool,
) -> Result<Prediction, String> {
    let wl = band_wavelength(band).ok_or_else(|| format!("unknown band '{band}'"))? / 1000.0;
    let b = prep
        .wavelengths
        .iter()
        .position(|&w| w == wl)
        .ok_or_else(|| format!("band '{band}' has no observations in this source"))?;
    let theta = prep.theta_from_hyper(hyper);
    let shifted: Vec<f64> = t.iter().map(|&x| x - prep.t_ref).collect();
    let (mean, var) = kalman::smooth(prep, &theta, &shifted, b);
    let noise = if include_noise { (hyper.jitter / prep.flux_scale).powi(2) } else { 0.0 };
    Ok(Prediction {
        mean: mean.iter().map(|m| m * prep.flux_scale).collect(),
        std: var.iter().map(|v| (v + noise).sqrt() * prep.flux_scale).collect(),
    })
}

// ---------------------------------------------------------------------------
// PyO3 Python bindings
// ---------------------------------------------------------------------------

#[cfg(feature = "python")]
mod python_bindings {
    use super::*;
    use features::{light_curve_features, BandFeatures, FeatureConfig};
    use thermal::{thermal_evolution, ThermalConfig, ThermalResult};
    use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyList};

    #[allow(clippy::too_many_arguments)]
    fn make_config(
        n_starts: usize,
        max_iters: usize,
        min_obs: usize,
        length_scale_bounds: Option<(f64, f64)>,
        wavelength_scale_prior: (f64, f64),
        jitter_prior: (f64, f64),
        laplace_errors: bool,
        use_upper_limits: bool,
    ) -> GpConfig {
        GpConfig {
            n_starts,
            max_iters,
            min_obs,
            length_scale_bounds,
            wavelength_scale_prior,
            jitter_prior,
            laplace_errors,
            use_upper_limits,
            ..GpConfig::default()
        }
    }

    struct Fitted {
        obs: Vec<Obs>,
        result: GpResult,
        features: Option<Vec<BandFeatures>>,
        thermal: Option<ThermalResult>,
    }

    /// Load, fit and (optionally) extract features and temperatures for one CSV.
    fn fit_one(
        path: &str,
        config: &GpConfig,
        fcfg: Option<&FeatureConfig>,
        tcfg: Option<&ThermalConfig>,
    ) -> Result<Fitted, String> {
        let obs = load_csv_for(path, config)?;
        let prep = prepare(&obs, config)?;
        let result = fit_prepared(&prep, config);
        let features = fcfg.map(|f| light_curve_features(&prep, &result.hyper, f));
        let thermal = tcfg.map(|t| thermal_evolution(&prep, &result.hyper, t));
        Ok(Fitted { obs, result, features, thermal })
    }

    fn thermal_config(thermal: bool, n_samples: usize, seed: u64) -> Option<ThermalConfig> {
        thermal.then(|| ThermalConfig { n_samples, seed, ..ThermalConfig::default() })
    }

    fn feature_config(features: bool, n_samples: usize, seed: u64) -> Option<FeatureConfig> {
        features.then(|| FeatureConfig { n_samples, seed, ..FeatureConfig::default() })
    }

    /// Fit a multi-band Matérn-3/2 GP to a ZTF/LSST photometry CSV.
    ///
    /// Returns a dict with keys:
    ///   params         – length_scale (days), wavelength_scale (Å), jitter and
    ///                    amp_<band> (flux), plus <name>_ln_err Laplace errors
    ///                    on ln(param) (None if unavailable)
    ///   features       – {band: {peak_time, peak_mag, peak_flux, t_rise, t_fade,
    ///                    fwhm, rise_rate, fade_rate, <name>_err, peak_bracketed,
    ///                    n_points}} (only with features=True; see features())
    ///   thermal        – blackbody temperature evolution, log_temp_peak and
    ///                    cooling_rate (only with thermal=True; see thermal())
    ///   obs            – list of {time, flux, flux_err, band, upper_limit}
    ///                    (physical flux; limits are flux 0 ± f_lim/5)
    ///   log_likelihood, log_posterior, n_obs (detections), n_limits, n_bands, n_starts,
    ///   n_starts_agree, converged
    #[pyfunction(name = "fit")]
    #[pyo3(signature = (
        csv_path, n_starts=3, max_iters=200, min_obs=5, length_scale_bounds=None,
        wavelength_scale_prior=(6000.0, 0.75), jitter_prior=(0.01, 2.0), laplace_errors=true,
        upper_limits=true, features=true, thermal=true, n_samples=100, seed=0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn py_fit(
        py: Python<'_>,
        csv_path: &str,
        n_starts: usize,
        max_iters: usize,
        min_obs: usize,
        length_scale_bounds: Option<(f64, f64)>,
        wavelength_scale_prior: (f64, f64),
        jitter_prior: (f64, f64),
        laplace_errors: bool,
        upper_limits: bool,
        features: bool,
        thermal: bool,
        n_samples: usize,
        seed: u64,
    ) -> PyResult<PyObject> {
        let config = make_config(
            n_starts, max_iters, min_obs, length_scale_bounds, wavelength_scale_prior, jitter_prior, laplace_errors,
            upper_limits,
        );
        let fcfg = feature_config(features, n_samples, seed);
        let tcfg = thermal_config(thermal, n_samples, seed);
        let fitted = py
            .allow_threads(|| fit_one(csv_path, &config, fcfg.as_ref(), tcfg.as_ref()))
            .map_err(PyRuntimeError::new_err)?;
        Ok(fitted_to_dict(py, &fitted)?.into_any().unbind())
    }

    /// Fit many CSVs in parallel across CPU cores (Rayon). Returns one dict per
    /// path, in input order; a path that fails gets {"error": "..."}.
    /// `n_threads=None` uses all cores. Takes the same keywords as fit().
    #[pyfunction]
    #[pyo3(signature = (
        csv_paths, n_threads=None, n_starts=3, max_iters=200, min_obs=5, length_scale_bounds=None,
        wavelength_scale_prior=(6000.0, 0.75), jitter_prior=(0.01, 2.0), laplace_errors=true,
        upper_limits=true, features=true, thermal=true, n_samples=100, seed=0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn fit_many(
        py: Python<'_>,
        csv_paths: Vec<String>,
        n_threads: Option<usize>,
        n_starts: usize,
        max_iters: usize,
        min_obs: usize,
        length_scale_bounds: Option<(f64, f64)>,
        wavelength_scale_prior: (f64, f64),
        jitter_prior: (f64, f64),
        laplace_errors: bool,
        upper_limits: bool,
        features: bool,
        thermal: bool,
        n_samples: usize,
        seed: u64,
    ) -> PyResult<PyObject> {
        let config = make_config(
            n_starts, max_iters, min_obs, length_scale_bounds, wavelength_scale_prior, jitter_prior, laplace_errors,
            upper_limits,
        );
        let fcfg = feature_config(features, n_samples, seed);
        let tcfg = thermal_config(thermal, n_samples, seed);
        let work = || -> Vec<Result<Fitted, String>> {
            csv_paths.par_iter().map(|p| fit_one(p, &config, fcfg.as_ref(), tcfg.as_ref())).collect()
        };
        let results = py.allow_threads(|| -> Result<_, String> {
            match n_threads {
                Some(n) => {
                    let pool = rayon::ThreadPoolBuilder::new().num_threads(n).build().map_err(|e| e.to_string())?;
                    Ok(pool.install(work))
                }
                None => Ok(work()),
            }
        });
        let results = results.map_err(PyRuntimeError::new_err)?;

        let out = PyList::empty(py);
        for r in &results {
            match r {
                Ok(fitted) => out.append(fitted_to_dict(py, fitted)?)?,
                Err(e) => {
                    let d = PyDict::new(py);
                    d.set_item("error", e.as_str())?;
                    out.append(d)?;
                }
            }
        }
        Ok(out.into_any().unbind())
    }

    /// Posterior mean and std of the GP light curve in `band` at `t_dense`.
    ///
    /// `result` is a dict returned by `fit()` / `fit_many()` (it needs both
    /// `params` and `obs`: a GP prediction conditions on the data). Returns
    /// `(mean, std)` lists in the same flux units as `result["obs"]`.
    #[pyfunction(name = "predict")]
    #[pyo3(signature = (result, t_dense, band, include_noise=false))]
    fn py_predict(
        py: Python<'_>,
        result: &Bound<'_, PyDict>,
        t_dense: Vec<f64>,
        band: &str,
        include_noise: bool,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let (prep, hyper) = rebuild(result)?;
        let pred = py
            .allow_threads(|| super::predict(&prep, &hyper, &t_dense, band, include_noise))
            .map_err(PyValueError::new_err)?;
        Ok((pred.mean, pred.std))
    }

    /// Per-band light-curve features from a `fit()` result:
    ///   peak_time, peak_mag (AB, ZP 23.9), peak_flux,
    ///   t_rise / t_fade – days from half-max to peak / peak to half-max,
    ///   fwhm            – t_rise + t_fade (days),
    ///   rise_rate / fade_rate – 2.5·log10(2) / t_rise, / t_fade (mag/day),
    ///   <name>_err      – robust 1σ from `n_samples` posterior draws,
    ///   peak_bracketed  – peak strictly inside the band's detections,
    ///   n_points.
    /// Each band is searched only between its first and last detection; a
    /// feature whose half-max crossing is not observed is None.
    #[pyfunction(name = "features")]
    #[pyo3(signature = (result, n_samples=100, seed=0, min_band_points=3))]
    fn py_features(
        py: Python<'_>,
        result: &Bound<'_, PyDict>,
        n_samples: usize,
        seed: u64,
        min_band_points: usize,
    ) -> PyResult<PyObject> {
        let (prep, hyper) = rebuild(result)?;
        let cfg = FeatureConfig { n_samples, seed, min_band_points, ..FeatureConfig::default() };
        let feats = py.allow_threads(|| light_curve_features(&prep, &hyper, &cfg));
        Ok(features_to_dict(py, &feats)?.into_any().unbind())
    }

    /// Blackbody temperature evolution from a `fit()` result (observed frame):
    ///   times, log_temp, log_temp_err, chi2   – per-epoch lists; bands – SED bands
    ///   ref_band, peak_time, log_temp_peak(_err)          – T at the light-curve peak
    ///   cooling_rate(_err) – d log10 T / dt over [peak, peak + cooling_window] (dex/day, <0 = cooling)
    ///   log_temp_latest(_err)                             – T at the last usable epoch
    #[pyfunction(name = "thermal")]
    #[pyo3(signature = (result, n_samples=100, seed=0, min_snr=3.0, min_band_points=3, cooling_window=Some(30.0)))]
    fn py_thermal(
        py: Python<'_>,
        result: &Bound<'_, PyDict>,
        n_samples: usize,
        seed: u64,
        min_snr: f64,
        min_band_points: usize,
        cooling_window: Option<f64>,
    ) -> PyResult<PyObject> {
        let (prep, hyper) = rebuild(result)?;
        let cfg = ThermalConfig { n_samples, seed, min_snr, min_band_points, cooling_window, ..ThermalConfig::default() };
        let th = py.allow_threads(|| thermal_evolution(&prep, &hyper, &cfg));
        Ok(thermal_to_dict(py, &th)?.into_any().unbind())
    }

    fn thermal_to_dict<'py>(py: Python<'py>, th: &ThermalResult) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("times", &th.times)?;
        d.set_item("log_temp", &th.log_temp)?;
        d.set_item("log_temp_err", &th.log_temp_err)?;
        d.set_item("chi2", &th.chi2)?;
        d.set_item("bands", &th.bands)?;
        d.set_item("ref_band", &th.ref_band)?;
        d.set_item("peak_time", th.peak_time)?;
        d.set_item("log_temp_peak", th.log_temp_peak)?;
        d.set_item("log_temp_peak_err", th.log_temp_peak_err)?;
        d.set_item("cooling_rate", th.cooling_rate)?;
        d.set_item("cooling_rate_err", th.cooling_rate_err)?;
        d.set_item("log_temp_latest", th.log_temp_latest)?;
        d.set_item("log_temp_latest_err", th.log_temp_latest_err)?;
        Ok(d)
    }

    fn fitted_to_dict<'py>(py: Python<'py>, f: &Fitted) -> PyResult<Bound<'py, PyDict>> {
        let d = result_to_dict(py, &f.result, &f.obs, f.features.as_deref())?;
        if let Some(th) = &f.thermal {
            d.set_item("thermal", thermal_to_dict(py, th)?)?;
        }
        Ok(d)
    }

    /// Reconstruct the prepared source and hyperparameters from a result dict.
    fn rebuild(result: &Bound<'_, PyDict>) -> PyResult<(PreparedSource, GpHyper)> {
        let params = get(result, "params")?.downcast_into::<PyDict>()?;
        let obs_list = get(result, "obs")?.downcast_into::<PyList>()?;
        let mut obs = Vec::with_capacity(obs_list.len());
        for item in obs_list.iter() {
            let d = item.downcast_into::<PyDict>()?;
            obs.push(Obs {
                time: get(&d, "time")?.extract()?,
                flux: get(&d, "flux")?.extract()?,
                flux_err: get(&d, "flux_err")?.extract()?,
                band: get(&d, "band")?.extract()?,
                upper_limit: match d.get_item("upper_limit")? {
                    Some(v) => v.extract()?,
                    None => false,
                },
            });
        }
        let config = GpConfig { min_obs: 1, ..GpConfig::default() };
        let prep = prepare(&obs, &config).map_err(PyValueError::new_err)?;
        let hyper = GpHyper {
            length_scale: get(&params, "length_scale")?.extract()?,
            wavelength_scale: get(&params, "wavelength_scale")?.extract()?,
            jitter: get(&params, "jitter")?.extract()?,
            amplitudes: prep
                .bands
                .iter()
                .map(|b| Ok((b.clone(), get(&params, &format!("amp_{b}"))?.extract()?)))
                .collect::<PyResult<_>>()?,
        };
        Ok((prep, hyper))
    }

    fn get<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
        d.get_item(key)?.ok_or_else(|| PyKeyError::new_err(key.to_string()))
    }

    fn features_to_dict<'py>(py: Python<'py>, feats: &[BandFeatures]) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for f in feats {
            let d = PyDict::new(py);
            let fields: [(&str, Option<f64>); 15] = [
                ("peak_time", f.peak_time),
                ("peak_mag", f.peak_mag),
                ("peak_flux", f.peak_flux),
                ("t_rise", f.t_rise),
                ("t_fade", f.t_fade),
                ("fwhm", f.fwhm),
                ("rise_rate", f.rise_rate),
                ("fade_rate", f.fade_rate),
                ("peak_time_err", f.peak_time_err),
                ("peak_mag_err", f.peak_mag_err),
                ("t_rise_err", f.t_rise_err),
                ("t_fade_err", f.t_fade_err),
                ("fwhm_err", f.fwhm_err),
                ("rise_rate_err", f.rise_rate_err),
                ("fade_rate_err", f.fade_rate_err),
            ];
            for (k, v) in fields {
                d.set_item(k, v)?;
            }
            d.set_item("peak_bracketed", f.peak_bracketed)?;
            d.set_item("n_points", f.n_points)?;
            out.set_item(f.band.as_str(), d)?;
        }
        Ok(out)
    }

    fn result_to_dict<'py>(
        py: Python<'py>,
        r: &GpResult,
        obs: &[Obs],
        feats: Option<&[BandFeatures]>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        let params = PyDict::new(py);
        let mut values = vec![r.hyper.length_scale, r.hyper.wavelength_scale, r.hyper.jitter];
        values.extend(r.hyper.amplitudes.iter().map(|(_, a)| *a));
        for (i, (name, v)) in r.param_names.iter().zip(&values).enumerate() {
            params.set_item(name, v)?;
            match &r.ln_err {
                Some(e) => params.set_item(format!("{name}_ln_err"), e[i])?,
                None => params.set_item(format!("{name}_ln_err"), py.None())?,
            }
        }
        dict.set_item("params", params)?;
        if let Some(f) = feats {
            dict.set_item("features", features_to_dict(py, f)?)?;
        }
        dict.set_item("log_likelihood", r.log_likelihood)?;
        dict.set_item("log_posterior", r.log_posterior)?;
        dict.set_item("n_obs", r.n_obs)?;
        dict.set_item("n_limits", r.n_limits)?;
        dict.set_item("n_bands", r.n_bands)?;
        dict.set_item("n_starts", r.n_starts)?;
        dict.set_item("n_starts_agree", r.n_starts_agree)?;
        dict.set_item("converged", r.converged)?;

        let obs_list = PyList::empty(py);
        for o in obs {
            let d = PyDict::new(py);
            d.set_item("time", o.time)?;
            d.set_item("flux", o.flux)?;
            d.set_item("flux_err", o.flux_err)?;
            d.set_item("band", o.band.as_str())?;
            d.set_item("upper_limit", o.upper_limit)?;
            obs_list.append(d)?;
        }
        dict.set_item("obs", obs_list)?;
        Ok(dict)
    }

    #[pymodule]
    fn rust_gp(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(py_fit, m)?)?;
        m.add_function(wrap_pyfunction!(fit_many, m)?)?;
        m.add_function(wrap_pyfunction!(py_predict, m)?)?;
        m.add_function(wrap_pyfunction!(py_features, m)?)?;
        m.add_function(wrap_pyfunction!(py_thermal, m)?)?;
        Ok(())
    }
}
