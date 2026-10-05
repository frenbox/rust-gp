//! Exact draws from the GP prior, for tests and benchmarks.
//!
//! Simulates the state-space model directly (O(n)), so the draws come from
//! precisely the model the fitter assumes — which is what a recovery test
//! needs.

use crate::rng::Rng;
use crate::{band_wavelength, kalman, linalg, GpHyper, Obs, IDX_AMP, IDX_JIT, IDX_LS, IDX_WL};

/// What to simulate.
#[derive(Debug, Clone)]
pub struct SimSpec {
    pub n_obs: usize,
    /// Time span; epochs are uniform on `[t_start, t_start + baseline]`.
    pub baseline: f64,
    pub t_start: f64,
    /// True hyperparameters. `amplitudes` also fixes which bands are drawn
    /// (each point picks one uniformly). `jitter` is extra scatter that is
    /// *not* reflected in the reported `flux_err`.
    pub hyper: GpHyper,
    /// Reported (and actual) per-point measurement error.
    pub flux_err: f64,
}

impl SimSpec {
    /// A two-band (g, r) default resembling a normalised transient.
    pub fn two_band(n_obs: usize, baseline: f64) -> Self {
        SimSpec {
            n_obs,
            baseline,
            t_start: 2_460_000.5,
            hyper: GpHyper {
                length_scale: 8.0,
                wavelength_scale: 6000.0,
                jitter: 0.01,
                amplitudes: vec![("g".into(), 0.8), ("r".into(), 1.0)],
            },
            flux_err: 0.05,
        }
    }
}

/// Draw one light curve from the GP prior.
pub fn simulate(spec: &SimSpec, seed: u64) -> Vec<Obs> {
    let mut rng = Rng::new(seed);
    let bands: Vec<&str> = spec.hyper.amplitudes.iter().map(|(b, _)| b.as_str()).collect();
    let wls: Vec<f64> = bands.iter().map(|b| band_wavelength(b).expect("unknown band") / 1000.0).collect();
    let mut theta = vec![0.0; IDX_AMP + bands.len()];
    theta[IDX_LS] = spec.hyper.length_scale.ln();
    theta[IDX_WL] = (spec.hyper.wavelength_scale / 1000.0).ln();
    theta[IDX_JIT] = spec.hyper.jitter.max(1e-12).ln();
    for (i, (_, a)) in spec.hyper.amplitudes.iter().enumerate() {
        theta[IDX_AMP + i] = a.ln();
    }

    let mut times: Vec<f64> = (0..spec.n_obs).map(|_| rng.uniform() * spec.baseline).collect();
    times.sort_by(f64::total_cmp);

    let d = 2 * bands.len();
    let draw = |cov: &[f64], rng: &mut Rng| -> Vec<f64> {
        let l = linalg::cholesky_regularised(cov, d).expect("covariance not finite");
        let z: Vec<f64> = (0..d).map(|_| rng.normal()).collect();
        (0..d).map(|i| (0..=i).map(|k| l[i * d + k] * z[k]).sum()).collect()
    };

    let (_, p_inf) = kalman::transition_and_noise(&theta, &wls, None);
    let mut x = draw(&p_inf, &mut rng);
    let mut obs = Vec::with_capacity(spec.n_obs);
    let mut t_prev = times.first().copied().unwrap_or(0.0);
    for &t in &times {
        let dt = t - t_prev;
        t_prev = t;
        if dt > 0.0 {
            let (a, q) = kalman::transition_and_noise(&theta, &wls, Some(dt));
            let w = draw(&q, &mut rng);
            x = (0..d).map(|i| (0..d).map(|j| a[i * d + j] * x[j]).sum::<f64>() + w[i]).collect();
        }
        let b = ((rng.uniform() * bands.len() as f64) as usize).min(bands.len() - 1);
        let noise = spec.flux_err * rng.normal() + spec.hyper.jitter * rng.normal();
        obs.push(Obs { time: spec.t_start + t, flux: x[2 * b] + noise, flux_err: spec.flux_err, band: bands[b].to_string(), upper_limit: false });
    }
    obs
}
