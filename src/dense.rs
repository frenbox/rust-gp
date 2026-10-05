//! Dense O(n³) reference implementation of the same GP.
//!
//! Not used by the fitter. It exists so tests and `likelihood-bench` can check
//! that the Kalman likelihood and smoother agree with the textbook
//! `K⁻¹`-based formulas, and to time the two against each other.

use crate::linalg;
use crate::{PreparedSource, IDX_AMP, IDX_JIT, IDX_LS, IDX_WL};

const LN_2PI: f64 = 1.837_877_066_409_345_5;

fn kernel(prep: &PreparedSource, theta: &[f64], t1: f64, b1: usize, t2: f64, b2: usize) -> f64 {
    let ell = theta[IDX_LS].exp();
    let ell_wl = theta[IDX_WL].exp();
    let dl = prep.wavelengths[b1] - prep.wavelengths[b2];
    let c = (theta[IDX_AMP + b1] + theta[IDX_AMP + b2] - 0.5 * dl * dl / (ell_wl * ell_wl)).exp();
    let x = 3f64.sqrt() * (t1 - t2).abs() / ell;
    c * (1.0 + x) * (-x).exp()
}

/// Cholesky factor of `K + diag(σ_i² + jitter²)` over the observations.
fn factor(prep: &PreparedSource, theta: &[f64]) -> Option<Vec<f64>> {
    let n = prep.times.len();
    let jit2 = (2.0 * theta[IDX_JIT]).exp();
    let mut k = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let v = kernel(prep, theta, prep.times[i], prep.band_idx[i], prep.times[j], prep.band_idx[j]);
            k[i * n + j] = v;
            k[j * n + i] = v;
        }
        k[i * n + i] += prep.flux_err[i] * prep.flux_err[i] + jit2;
    }
    linalg::cholesky(&mut k, n).then_some(k)
}

/// `ln p(y | θ)` by dense Cholesky.
pub fn log_likelihood(prep: &PreparedSource, theta: &[f64]) -> f64 {
    let n = prep.times.len();
    let Some(l) = factor(prep, theta) else { return f64::NAN };
    let mut alpha = prep.flux.clone();
    linalg::chol_solve(&l, n, &mut alpha);
    let quad: f64 = prep.flux.iter().zip(&alpha).map(|(y, a)| y * a).sum();
    let logdet: f64 = (0..n).map(|i| l[i * n + i].ln()).sum::<f64>() * 2.0;
    -0.5 * (quad + logdet + n as f64 * LN_2PI)
}

/// Posterior mean and variance of the latent `f_band` at shifted times.
pub fn predict(prep: &PreparedSource, theta: &[f64], t_query: &[f64], band: usize) -> (Vec<f64>, Vec<f64>) {
    let n = prep.times.len();
    let l = factor(prep, theta).expect("kernel matrix not positive definite");
    let mut alpha = prep.flux.clone();
    linalg::chol_solve(&l, n, &mut alpha);
    let mut mean = Vec::with_capacity(t_query.len());
    let mut var = Vec::with_capacity(t_query.len());
    for &t in t_query {
        let ks: Vec<f64> = (0..n)
            .map(|i| kernel(prep, theta, t, band, prep.times[i], prep.band_idx[i]))
            .collect();
        mean.push(ks.iter().zip(&alpha).map(|(a, b)| a * b).sum());
        let mut v = ks.clone();
        linalg::chol_solve(&l, n, &mut v);
        let kss = kernel(prep, theta, t, band, t, band);
        var.push(kss - ks.iter().zip(&v).map(|(a, b)| a * b).sum::<f64>());
    }
    (mean, var)
}
