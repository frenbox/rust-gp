//! Blackbody temperature evolution from the multi-band GP.
//!
//! The GP is already two-dimensional — time × wavelength through the band
//! kernel `exp(−Δλ² / 2ℓ_λ²)` — so the posterior gives every band's flux at
//! every epoch, jointly. At each epoch the SED `{f_b}` is fitted with
//!
//!   f_ν(λ_b) = A · B_ν(λ_b, T),   B_ν ∝ λ⁻³ / (exp(hc / λkT) − 1)
//!
//! weighted by the posterior variances, with `A` profiled out analytically.
//!
//! Reported (all temperatures are **observed-frame**, no redshift or
//! extinction correction):
//!
//! * per-epoch `log_temp` (log₁₀ K) with 1σ errors;
//! * `log_temp_peak` — at the light-curve peak of the reference band (the
//!   band with the most detections whose peak is bracketed; ties go to the
//!   band nearest r);
//! * `cooling_rate` — least-squares slope d log₁₀T / dt (dex/day) over the
//!   epochs in `[t_peak, t_peak + cooling_window]` (30 d by default: the
//!   photospheric phase, where supernovae cool and TDEs stay hot); negative
//!   means cooling. Same units and sign as boom-astro/lightcurve-fitting's
//!   `thermal_cooling_rate`, which instead fits all epochs;
//! * `log_temp_latest` — at the last usable epoch.
//!
//! Uncertainties repeat the blackbody fits on joint posterior draws of the
//! light curve (which carry the cross-band and cross-epoch correlations),
//! conditional on the fitted GP hyperparameters.
//!
//! **One band set per source.** Real transient SEDs are not blackbodies (line
//! blanketing suppresses g in supernovae), so g−r and r−i imply different
//! temperatures, and letting bands drop in and out of the fit makes T(t)
//! jump. Bands with ≥ `min_band_points` detections form the set; an epoch is
//! used only if *every* band in it lies inside its own detection range and
//! is detected by the GP at `min_snr`. If that leaves fewer than
//! `min_epochs` epochs, the two-band subset with the most epochs is used
//! instead; `bands` reports the choice. With two bands the
//! fit is exact (one colour, two unknowns), so χ² is 0 and only the
//! posterior spread constrains `T`. ZTF g/r/i span 4770–7625 Å, i.e. the
//! Rayleigh–Jeans side for T ≳ 15 000 K, where colours lose leverage and the
//! errors grow accordingly; fits that run into the search bounds are dropped.

use serde::{Deserialize, Serialize};

use crate::features::{light_curve_features, FeatureConfig};
use crate::rng::Rng;
use crate::{kalman, GpHyper, PreparedSource};

/// hc / k_B in Å·K.
const HC_OVER_K: f64 = 1.438_776_877e8;
/// Search range for log₁₀ T (K).
const LOG_T_RANGE: (f64, f64) = (3.3, 5.0);

/// Settings for [`thermal_evolution`].
#[derive(Debug, Clone)]
pub struct ThermalConfig {
    /// Posterior draws for the uncertainties (0 = values only).
    pub n_samples: usize,
    pub seed: u64,
    /// A band joins an epoch's SED only if the GP mean / σ reaches this.
    pub min_snr: f64,
    /// Detections a band needs to join the SED.
    pub min_band_points: usize,
    /// Fewest epochs before falling back from all bands to the best pair.
    pub min_epochs: usize,
    /// Post-peak window (days) for the cooling-rate fit; `None` = all epochs.
    pub cooling_window: Option<f64>,
    /// Cap on epochs (spacing is min(1 d, ℓ/10) across the detections).
    pub max_epochs: usize,
}

impl Default for ThermalConfig {
    fn default() -> Self {
        ThermalConfig {
            n_samples: 100,
            seed: 0,
            min_snr: 3.0,
            min_band_points: 3,
            min_epochs: 5,
            cooling_window: Some(30.0),
            max_epochs: 400,
        }
    }
}

/// Temperature evolution of one source.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThermalResult {
    /// Epochs (input time system) with a usable SED.
    pub times: Vec<f64>,
    /// log₁₀ T (K) at each epoch, from the posterior mean SED.
    pub log_temp: Vec<f64>,
    pub log_temp_err: Vec<Option<f64>>,
    /// Weighted χ² of the blackbody fit (0 with two bands).
    pub chi2: Vec<f64>,
    /// Bands in every SED (constant across epochs).
    pub bands: Vec<String>,
    /// Band whose peak defines `peak_time`.
    pub ref_band: Option<String>,
    pub peak_time: Option<f64>,
    pub log_temp_peak: Option<f64>,
    pub log_temp_peak_err: Option<f64>,
    /// d log₁₀T / dt over post-peak epochs (dex/day); negative = cooling.
    pub cooling_rate: Option<f64>,
    pub cooling_rate_err: Option<f64>,
    pub log_temp_latest: Option<f64>,
    pub log_temp_latest_err: Option<f64>,
}

/// Planck `B_ν` up to a constant: `λ⁻³ / expm1(hc / λkT)`, λ in units of 1000 Å.
#[inline]
fn planck(lambda_kaa: f64, temp: f64) -> f64 {
    let x = HC_OVER_K / (lambda_kaa * 1000.0 * temp);
    lambda_kaa.powi(-3) / x.exp_m1()
}

/// Weighted blackbody fit to fluxes at wavelengths `lambda_kaa` (1000 Å).
/// Returns `(log10 T, amplitude, chi2)`, or `None` if the best fit is at the
/// edge of the temperature range or has a non-positive amplitude.
pub fn fit_blackbody(lambda_kaa: &[f64], flux: &[f64], var: &[f64]) -> Option<(f64, f64, f64)> {
    if lambda_kaa.len() < 2 {
        return None;
    }
    let w: Vec<f64> = var.iter().map(|v| 1.0 / v.max(1e-300)).collect();
    // χ²(log T) with the amplitude profiled out (weighted linear LS).
    let profile = |log_t: f64| -> (f64, f64) {
        let t = 10f64.powf(log_t);
        let (mut sfb, mut sbb) = (0.0, 0.0);
        let b: Vec<f64> = lambda_kaa.iter().map(|&l| planck(l, t)).collect();
        for i in 0..b.len() {
            sfb += w[i] * flux[i] * b[i];
            sbb += w[i] * b[i] * b[i];
        }
        let a = sfb / sbb;
        let chi2 = (0..b.len()).map(|i| w[i] * (flux[i] - a * b[i]).powi(2)).sum();
        (chi2, a)
    };

    // Coarse scan, then golden-section refinement around the best cell.
    let (lo, hi) = LOG_T_RANGE;
    let n = 86;
    let step = (hi - lo) / (n - 1) as f64;
    let ibest = (0..n)
        .min_by(|&a, &b| profile(lo + a as f64 * step).0.total_cmp(&profile(lo + b as f64 * step).0))?;
    if ibest == 0 || ibest == n - 1 {
        return None;
    }
    let (mut a, mut b) = (lo + (ibest - 1) as f64 * step, lo + (ibest + 1) as f64 * step);
    let g = 0.5 * (5f64.sqrt() - 1.0);
    let (mut c, mut d) = (b - g * (b - a), a + g * (b - a));
    let (mut fc, mut fd) = (profile(c).0, profile(d).0);
    for _ in 0..40 {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - g * (b - a);
            fc = profile(c).0;
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + g * (b - a);
            fd = profile(d).0;
        }
    }
    let log_t = 0.5 * (a + b);
    let (chi2, amp) = profile(log_t);
    (amp > 0.0 && chi2.is_finite()).then_some((log_t, amp, chi2))
}

/// Ordinary least-squares slope of `y` on `x`.
fn ols_slope(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() < 3 {
        return None;
    }
    let n = x.len() as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let sxx: f64 = x.iter().map(|v| (v - mx).powi(2)).sum();
    let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    (sxx > 1e-12).then(|| sxy / sxx)
}

fn robust_std(mut v: Vec<f64>) -> Option<f64> {
    if v.len() < 10 {
        return None;
    }
    v.sort_by(f64::total_cmp);
    let q = |p: f64| {
        let x = p * (v.len() - 1) as f64;
        let (i, w) = (x.floor() as usize, x.fract());
        v[i] * (1.0 - w) + v[(i + 1).min(v.len() - 1)] * w
    };
    Some(0.5 * (q(0.84) - q(0.16)))
}

/// Temperature evolution, peak temperature and cooling rate of a fitted source.
pub fn thermal_evolution(prep: &PreparedSource, hyper: &GpHyper, cfg: &ThermalConfig) -> ThermalResult {
    let mut out = ThermalResult::default();
    let nb = prep.n_bands();

    // Per-band detection ranges (shifted times) and counts.
    let mut ranges = vec![(f64::INFINITY, f64::NEG_INFINITY); nb];
    let mut counts = vec![0usize; nb];
    for i in 0..prep.n_obs() {
        if !prep.upper_limit[i] {
            let b = prep.band_idx[i];
            ranges[b].0 = ranges[b].0.min(prep.times[i]);
            ranges[b].1 = ranges[b].1.max(prep.times[i]);
            counts[b] += 1;
        }
    }
    let eligible: Vec<usize> = (0..nb).filter(|&b| counts[b] >= cfg.min_band_points).collect();
    if eligible.len() < 2 {
        return out;
    }
    let first = ranges.iter().map(|r| r.0).fold(f64::INFINITY, f64::min);
    let last = ranges.iter().map(|r| r.1).fold(f64::NEG_INFINITY, f64::max);
    let span = (last - first).max(0.0);
    let step = (hyper.length_scale / 10.0).min(1.0).max(span / cfg.max_epochs.max(2) as f64).max(1e-3);
    let n_ep = (span / step).floor() as usize + 1;
    let grid: Vec<f64> = (0..n_ep).map(|i| first + i as f64 * step).collect();

    let theta = prep.theta_from_hyper(hyper);
    let (mean, var) = kalman::smooth_all(prep, &theta, &grid);

    // Grid points where every band of `set` is usable.
    let usable = |b: usize, q: usize| {
        let (lo, hi) = ranges[b];
        let t = grid[q];
        t >= lo && t <= hi && mean[b][q] > 0.0 && mean[b][q] >= cfg.min_snr * var[b][q].sqrt()
    };
    let epochs_for = |set: &[usize]| -> Vec<usize> {
        (0..grid.len()).filter(|&q| set.iter().all(|&b| usable(b, q))).collect()
    };
    // All eligible bands, else the pair with the most common epochs.
    let mut bands = eligible.clone();
    let mut qs = epochs_for(&bands);
    if qs.len() < cfg.min_epochs && bands.len() > 2 {
        let best_pair = eligible
            .iter()
            .enumerate()
            .flat_map(|(i, &a)| eligible[i + 1..].iter().map(move |&b| vec![a, b]))
            .map(|pair| {
                let q = epochs_for(&pair);
                (pair, q)
            })
            .max_by_key(|(_, q)| q.len());
        if let Some((pair, pq)) = best_pair.filter(|(_, pq)| pq.len() > qs.len()) {
            bands = pair;
            qs = pq;
        }
    }
    out.bands = bands.iter().map(|&b| prep.bands[b].clone()).collect();
    let lam: Vec<f64> = bands.iter().map(|&b| prep.wavelengths[b]).collect();

    // Posterior-mean SED fit at each epoch.
    struct Epoch {
        q: usize,
        log_t: f64,
    }
    let mut epochs: Vec<Epoch> = Vec::new();
    for &q in &qs {
        let f: Vec<f64> = bands.iter().map(|&b| mean[b][q]).collect();
        let v: Vec<f64> = bands.iter().map(|&b| var[b][q]).collect();
        if let Some((log_t, _, chi2)) = fit_blackbody(&lam, &f, &v) {
            out.times.push(grid[q] + prep.t_ref);
            out.log_temp.push(log_t);
            out.chi2.push(chi2);
            epochs.push(Epoch { q, log_t });
        }
    }
    if epochs.is_empty() {
        return out;
    }

    // Reference peak: the best-sampled band with a bracketed peak; ties go
    // to the band nearest r.
    let feats = light_curve_features(prep, hyper, &FeatureConfig { n_samples: 0, ..Default::default() });
    let dist_r = |band: &str| (crate::band_wavelength(band).unwrap_or(0.0) - 6231.0).abs();
    let peak = feats
        .iter()
        .filter(|f| f.peak_bracketed && f.peak_time.is_some())
        .max_by(|a, b| a.n_points.cmp(&b.n_points).then(dist_r(&b.band).total_cmp(&dist_r(&a.band))))
        .map(|f| (f.band.clone(), f.peak_time.unwrap() - prep.t_ref));
    // Epoch index nearest the peak, if one lies within a grid step of it.
    let peak_epoch = peak.as_ref().and_then(|(_, tp)| {
        let (i, d) = epochs
            .iter()
            .enumerate()
            .map(|(i, e)| (i, (grid[e.q] - tp).abs()))
            .min_by(|a, b| a.1.total_cmp(&b.1))?;
        (d <= step).then_some(i)
    });
    let post: Vec<usize> = match &peak {
        Some((_, tp)) => (0..epochs.len())
            .filter(|&i| {
                let t = grid[epochs[i].q];
                t >= *tp && cfg.cooling_window.is_none_or(|w| t <= tp + w)
            })
            .collect(),
        None => Vec::new(),
    };
    let cooling = |logs: &dyn Fn(usize) -> Option<f64>| -> Option<f64> {
        let (x, y): (Vec<f64>, Vec<f64>) =
            post.iter().filter_map(|&i| logs(i).map(|l| (grid[epochs[i].q], l))).unzip();
        // Require the post-peak SEDs to span at least two days.
        if x.len() < 3 || x.last()? - x.first()? < 2.0 {
            return None;
        }
        ols_slope(&x, &y)
    };

    out.ref_band = peak.as_ref().map(|(b, _)| b.clone());
    out.peak_time = peak.as_ref().map(|(_, tp)| tp + prep.t_ref);
    out.log_temp_peak = peak_epoch.map(|i| epochs[i].log_t);
    out.cooling_rate = cooling(&|i| Some(epochs[i].log_t));
    out.log_temp_latest = epochs.last().map(|e| e.log_t);

    // Uncertainties: refit every epoch on joint posterior draws.
    out.log_temp_err = vec![None; epochs.len()];
    if cfg.n_samples > 0 {
        let mut rng = Rng::new(cfg.seed);
        let draws = kalman::sample_posterior(prep, &theta, &grid, cfg.n_samples, &mut rng);
        let mut per_epoch: Vec<Vec<f64>> = vec![Vec::new(); epochs.len()];
        let (mut peaks, mut slopes, mut latest) = (Vec::new(), Vec::new(), Vec::new());
        for s in &draws {
            let fits: Vec<Option<f64>> = epochs
                .iter()
                .map(|e| {
                    let f: Vec<f64> = bands.iter().map(|&b| s[b][e.q]).collect();
                    let v: Vec<f64> = bands.iter().map(|&b| var[b][e.q]).collect();
                    fit_blackbody(&lam, &f, &v).map(|r| r.0)
                })
                .collect();
            for (i, l) in fits.iter().enumerate() {
                if let Some(l) = l {
                    per_epoch[i].push(*l);
                }
            }
            if let Some(Some(l)) = peak_epoch.map(|i| fits[i]) {
                peaks.push(l);
            }
            if let Some(Some(l)) = fits.last() {
                latest.push(*l);
            }
            if let Some(c) = cooling(&|i| fits[i]) {
                slopes.push(c);
            }
        }
        out.log_temp_err = per_epoch.into_iter().map(robust_std).collect();
        out.log_temp_peak_err = out.log_temp_peak.and(robust_std(peaks));
        out.cooling_rate_err = out.cooling_rate.and(robust_std(slopes));
        out.log_temp_latest_err = out.log_temp_latest.and(robust_std(latest));
    }
    out
}
