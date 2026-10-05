//! Compact unconstrained L-BFGS with a backtracking Armijo line search.
//!
//! Every hyperparameter is optimised in log space and kept in a sensible range
//! by soft priors, so no box constraints are needed. Written in-crate rather
//! than via `argmin` because a whole fit is ~10²–10³ µs and per-iteration
//! allocation and state cloning would be a visible fraction of that.

/// Stopping and memory settings.
#[derive(Debug, Clone, Copy)]
pub struct LbfgsOptions {
    pub max_iters: usize,
    /// Stop when `max |∂f/∂x| < grad_tol`.
    pub grad_tol: f64,
    /// Stop when one step reduces `f` by less than `f_tol · (1 + |f|)`.
    pub f_tol: f64,
    /// Number of correction pairs kept.
    pub memory: usize,
    /// Largest allowed change of any coordinate in one step (log units).
    pub max_step: f64,
}

impl Default for LbfgsOptions {
    fn default() -> Self {
        LbfgsOptions { max_iters: 200, grad_tol: 1e-6, f_tol: 1e-12, memory: 8, max_step: 2.0 }
    }
}

#[derive(Debug, Clone)]
pub struct LbfgsResult {
    pub x: Vec<f64>,
    pub f: f64,
    pub n_iters: usize,
    pub n_evals: usize,
    pub converged: bool,
}

/// Minimise `f`, where `f(x, grad)` returns the value and writes the gradient.
/// A non-finite value is treated as +∞ and makes the line search back off.
pub fn minimize<F>(mut f: F, x0: Vec<f64>, opts: &LbfgsOptions) -> LbfgsResult
where
    F: FnMut(&[f64], &mut [f64]) -> f64,
{
    let n = x0.len();
    let mut x = x0;
    let mut g = vec![0.0; n];
    let mut fx = f(&x, &mut g);
    let mut n_evals = 1;
    if !fx.is_finite() {
        return LbfgsResult { x, f: fx, n_iters: 0, n_evals, converged: false };
    }

    let mut s_hist: Vec<Vec<f64>> = Vec::with_capacity(opts.memory);
    let mut y_hist: Vec<Vec<f64>> = Vec::with_capacity(opts.memory);
    let mut rho_hist: Vec<f64> = Vec::with_capacity(opts.memory);
    let mut alpha = vec![0.0; opts.memory];
    let mut d = vec![0.0; n];
    let mut x_new = vec![0.0; n];
    let mut g_new = vec![0.0; n];
    let mut converged = false;
    let mut iters = 0;

    while iters < opts.max_iters {
        if inf_norm(&g) < opts.grad_tol {
            converged = true;
            break;
        }
        iters += 1;

        // Two-loop recursion: d = −H g.
        d.copy_from_slice(&g);
        let h = s_hist.len();
        for i in (0..h).rev() {
            alpha[i] = rho_hist[i] * dot(&s_hist[i], &d);
            axpy(-alpha[i], &y_hist[i], &mut d);
        }
        let gamma = if h > 0 {
            dot(&s_hist[h - 1], &y_hist[h - 1]) / dot(&y_hist[h - 1], &y_hist[h - 1])
        } else {
            1.0 / inf_norm(&g).max(1.0)
        };
        for v in &mut d {
            *v *= gamma;
        }
        for i in 0..h {
            let beta = rho_hist[i] * dot(&y_hist[i], &d);
            axpy(alpha[i] - beta, &s_hist[i], &mut d);
        }
        for v in &mut d {
            *v = -*v;
        }

        let mut slope = dot(&g, &d);
        if !(slope < 0.0) {
            // Not a descent direction: drop the curvature history.
            s_hist.clear();
            y_hist.clear();
            rho_hist.clear();
            let k = 1.0 / inf_norm(&g).max(1.0);
            for (di, gi) in d.iter_mut().zip(&g) {
                *di = -gi * k;
            }
            slope = dot(&g, &d);
        }
        let dn = inf_norm(&d);
        if dn > opts.max_step {
            let k = opts.max_step / dn;
            for v in &mut d {
                *v *= k;
            }
            slope *= k;
        }

        // Backtracking Armijo.
        let mut step = 1.0;
        let mut f_new = f64::INFINITY;
        let mut accepted = false;
        for _ in 0..40 {
            for i in 0..n {
                x_new[i] = x[i] + step * d[i];
            }
            f_new = f(&x_new, &mut g_new);
            n_evals += 1;
            if f_new.is_finite() && f_new <= fx + 1e-4 * step * slope {
                accepted = true;
                break;
            }
            // Quadratic interpolation of the step, safeguarded to [0.1, 0.5].
            let next = if f_new.is_finite() {
                let denom = 2.0 * (f_new - fx - step * slope);
                if denom > 0.0 { (-slope * step * step / denom).clamp(0.1 * step, 0.5 * step) } else { 0.5 * step }
            } else {
                0.25 * step
            };
            step = next;
        }
        if !accepted {
            // No progress possible along d: we are at a (numerical) minimum.
            converged = inf_norm(&g) < opts.grad_tol.sqrt();
            break;
        }

        let s: Vec<f64> = (0..n).map(|i| x_new[i] - x[i]).collect();
        let y: Vec<f64> = (0..n).map(|i| g_new[i] - g[i]).collect();
        let sy = dot(&s, &y);
        if sy > 1e-12 * dot(&y, &y).sqrt() * dot(&s, &s).sqrt() {
            if s_hist.len() == opts.memory {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            s_hist.push(s);
            y_hist.push(y);
            rho_hist.push(1.0 / sy);
        }

        let decrease = fx - f_new;
        x.copy_from_slice(&x_new);
        g.copy_from_slice(&g_new);
        fx = f_new;
        if decrease <= opts.f_tol * (1.0 + fx.abs()) {
            converged = true;
            break;
        }
    }

    LbfgsResult { x, f: fx, n_iters: iters, n_evals, converged }
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[inline]
fn axpy(a: f64, x: &[f64], y: &mut [f64]) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += a * xi;
    }
}

#[inline]
fn inf_norm(a: &[f64]) -> f64 {
    a.iter().fold(0.0, |m, v| m.max(v.abs()))
}
