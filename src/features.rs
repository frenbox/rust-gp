//! Light-curve shape features from the GP posterior, per band.
//!
//! Definitions (all in flux, then converted):
//!
//! * **peak** — time and AB magnitude (ZP 23.9) of maximum flux.
//! * **t_rise / t_fade** — time from the half-maximum crossing to the peak,
//!   and from the peak to the half-maximum crossing after it (days).
//! * **fwhm** — `t_rise + t_fade` (days).
//! * **rise_rate / fade_rate** — `2.5·log₁₀2 / t_rise` and `/ t_fade`
//!   (mag/day): the mean rate over the 0.753 mag between half-max and peak.
//!
//! Each band is searched only between its own first and last data point
//! (detection or non-detection, see [`crate::load_csv_with_limits`]): past
//! the data a zero-mean GP always decays to half-max eventually, so a crossing
//! found there would be an artefact. If the brightest point of the curve is
//! the band's first or last detection, the peak is not bracketed (it may lie
//! outside the data) and every feature is `None`; a rise or fade that is not
//! observed down to half-max is `None` on its own.
//!
//! Point values come from the posterior mean curve — the line that gets
//! plotted. Uncertainties are a robust standard deviation, (p84 − p16)/2, of
//! the same measurements on joint posterior draws of the curve
//! ([`kalman::sample_posterior`]); they are conditional on the fitted
//! hyperparameters.

use serde::{Deserialize, Serialize};

use crate::rng::Rng;
use crate::{kalman, GpHyper, PreparedSource};

/// Magnitude difference between peak and half-maximum, `2.5·log₁₀2`.
pub const HALF_MAX_MAG: f64 = 0.752_574_989_159_953_3;
const ZP: f64 = 23.9;

/// Settings for [`light_curve_features`].
#[derive(Debug, Clone)]
pub struct FeatureConfig {
    /// Posterior draws for the uncertainties (0 = values only).
    pub n_samples: usize,
    pub seed: u64,
    /// Bands with fewer detections get no features.
    pub min_band_points: usize,
    /// Cap on the evaluation grid (spacing is min(ℓ/20, span/500)).
    pub max_grid: usize,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        FeatureConfig { n_samples: 100, seed: 0, min_band_points: 3, max_grid: 20_000 }
    }
}

/// Features of one band. Times are in the input time system (JD/MJD).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BandFeatures {
    pub band: String,
    /// Detections in this band (non-detections not counted).
    pub n_points: usize,
    pub peak_time: Option<f64>,
    pub peak_mag: Option<f64>,
    pub peak_flux: Option<f64>,
    pub t_rise: Option<f64>,
    pub t_fade: Option<f64>,
    pub fwhm: Option<f64>,
    pub rise_rate: Option<f64>,
    pub fade_rate: Option<f64>,
    pub peak_time_err: Option<f64>,
    pub peak_mag_err: Option<f64>,
    pub t_rise_err: Option<f64>,
    pub t_fade_err: Option<f64>,
    pub fwhm_err: Option<f64>,
    pub rise_rate_err: Option<f64>,
    pub fade_rate_err: Option<f64>,
    /// The peak lies strictly inside the band's data range. When `false`
    /// (the brightest point is the first or last detection) all features are
    /// `None`.
    pub peak_bracketed: bool,
}

/// Shape measurements on one sampled curve.
#[derive(Debug, Clone, Copy)]
pub struct Shape {
    pub peak_time: f64,
    pub peak_flux: f64,
    pub t_rise: Option<f64>,
    pub t_fade: Option<f64>,
    /// Maximum is not at either end of the curve.
    pub interior: bool,
}

/// Measure peak and half-maximum widths of `f(t)` on an ascending grid.
/// `None` if the curve never goes above zero.
pub fn measure(t: &[f64], f: &[f64]) -> Option<Shape> {
    let n = t.len();
    let imax = (0..n).max_by(|&a, &b| f[a].total_cmp(&f[b]))?;
    if !(f[imax] > 0.0) {
        return None;
    }
    let interior = imax > 0 && imax + 1 < n;
    // Parabolic refinement of the peak between grid points.
    let (mut peak_time, mut peak_flux) = (t[imax], f[imax]);
    if interior {
        let (y0, y1, y2) = (f[imax - 1], f[imax], f[imax + 1]);
        let denom = y0 - 2.0 * y1 + y2;
        if denom < 0.0 {
            let off = (0.5 * (y0 - y2) / denom).clamp(-0.5, 0.5);
            let h = 0.5 * (t[imax + 1] - t[imax - 1]);
            peak_time += off * h;
            peak_flux = y1 - 0.25 * (y0 - y2) * off;
        }
    }
    let half = 0.5 * peak_flux;
    let cross = |i: usize, j: usize| t[i] + (half - f[i]) / (f[j] - f[i]) * (t[j] - t[i]);
    let t_rise = (0..imax).rev().find(|&i| f[i] < half).map(|i| peak_time - cross(i, i + 1));
    let t_fade = (imax + 1..n).find(|&i| f[i] < half).map(|i| cross(i - 1, i) - peak_time);
    Some(Shape { peak_time, peak_flux, t_rise, t_fade, interior })
}

fn flux_to_mag(f: f64) -> f64 {
    ZP - 2.5 * f.log10()
}

/// The seven reported quantities of one shape, in physical units.
fn quantities(s: &Shape, t_ref: f64, flux_scale: f64) -> [Option<f64>; 7] {
    let fwhm = match (s.t_rise, s.t_fade) {
        (Some(r), Some(f)) => Some(r + f),
        _ => None,
    };
    [
        Some(s.peak_time + t_ref),
        Some(flux_to_mag(s.peak_flux * flux_scale)),
        s.t_rise,
        s.t_fade,
        fwhm,
        s.t_rise.map(|r| HALF_MAX_MAG / r),
        s.t_fade.map(|f| HALF_MAX_MAG / f),
    ]
}

/// Robust spread (p84 − p16)/2 of the defined values; `None` below 10.
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

/// Peak, rise/fade times and rates, and FWHM for every band of a fitted source.
pub fn light_curve_features(prep: &PreparedSource, hyper: &GpHyper, cfg: &FeatureConfig) -> Vec<BandFeatures> {
    let theta = prep.theta_from_hyper(hyper);
    let span = prep.times.last().copied().unwrap_or(0.0).max(1e-6);
    let step = (hyper.length_scale / 20.0).min(span / 500.0);
    let n_grid = ((span / step).ceil() as usize + 1).clamp(2, cfg.max_grid);
    let grid: Vec<f64> = (0..n_grid).map(|i| span * i as f64 / (n_grid - 1) as f64).collect();

    let (mean, _) = kalman::smooth_all(prep, &theta, &grid);
    let samples = if cfg.n_samples > 0 {
        let mut rng = Rng::new(cfg.seed);
        kalman::sample_posterior(prep, &theta, &grid, cfg.n_samples, &mut rng)
    } else {
        Vec::new()
    };

    prep.bands
        .iter()
        .enumerate()
        .map(|(b, band)| {
            let in_band: Vec<usize> = (0..prep.n_obs()).filter(|&i| prep.band_idx[i] == b).collect();
            let n_det = in_band.iter().filter(|&&i| !prep.upper_limit[i]).count();
            let mut out = BandFeatures { band: band.clone(), n_points: n_det, ..Default::default() };
            if n_det < cfg.min_band_points.max(1) {
                return out;
            }
            // Grid points inside this band's data, non-detections included:
            // they bound the rise/fade, which is the reason to load them.
            let times: Vec<f64> = in_band.iter().map(|&i| prep.times[i]).collect();
            let (first, last) = (times[0], times[times.len() - 1]);
            let tol = 1e-9 * span.max(1.0);
            let lo = grid.partition_point(|&t| t < first - tol);
            let hi = grid.partition_point(|&t| t <= last + tol);
            if hi < lo + 2 {
                return out;
            }
            let g = &grid[lo..hi];
            let Some(shape) = measure(g, &mean[b][lo..hi]) else { return out };
            out.peak_bracketed = shape.interior;
            if !shape.interior {
                return out;
            }
            let v = quantities(&shape, prep.t_ref, prep.flux_scale);
            out.peak_flux = Some(shape.peak_flux * prep.flux_scale);
            [out.peak_time, out.peak_mag, out.t_rise, out.t_fade, out.fwhm, out.rise_rate, out.fade_rate] = v;

            if !samples.is_empty() {
                let mut cols: [Vec<f64>; 7] = Default::default();
                for s in &samples {
                    if let Some(sh) = measure(g, &s[b][lo..hi]).filter(|sh| sh.interior) {
                        for (col, q) in cols.iter_mut().zip(quantities(&sh, prep.t_ref, prep.flux_scale)) {
                            if let Some(q) = q.filter(|q| q.is_finite()) {
                                col.push(q);
                            }
                        }
                    }
                }
                let errs = cols.map(robust_std);
                // An error only makes sense next to a value.
                let keep = |val: Option<f64>, e: Option<f64>| val.and(e);
                out.peak_time_err = keep(out.peak_time, errs[0]);
                out.peak_mag_err = keep(out.peak_mag, errs[1]);
                out.t_rise_err = keep(out.t_rise, errs[2]);
                out.t_fade_err = keep(out.t_fade, errs[3]);
                out.fwhm_err = keep(out.fwhm, errs[4]);
                out.rise_rate_err = keep(out.rise_rate, errs[5]);
                out.fade_rate_err = keep(out.fade_rate, errs[6]);
            }
            out
        })
        .collect()
}
