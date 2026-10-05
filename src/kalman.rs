//! Exact state-space evaluation of the multi-band Matérn-3/2 GP.
//!
//! Kernel (intrinsic coregionalisation, as in avocado / PLAsTiCC):
//!
//!   k((t, b), (t', b')) = C[b, b'] · k₃/₂(|t − t'|; ℓ)
//!   C[b, b'] = σ_b σ_b' · exp(−(λ_b − λ_b')² / 2ℓ_λ²)
//!   k₃/₂(τ; ℓ) = (1 + √3 τ/ℓ) · exp(−√3 τ/ℓ)
//!
//! A unit Matérn-3/2 process has the exact two-dimensional state `[f, f']`
//! with stationary covariance `P∞ = diag(1, λ²)`, `λ = √3/ℓ`, and transition
//!
//!   A(Δ) = e^{−λΔ} [[1 + λΔ, Δ], [−λ²Δ, 1 − λΔ]],   Q(Δ) = P∞ − A P∞ Aᵀ.
//!
//! Stacking one such pair per band gives a `2B`-dimensional state with
//! transition `I_B ⊗ A(Δ)` and process noise `C ⊗ Q(Δ)`; each observation
//! reads off the `f` component of its band. The Kalman filter on this model
//! reproduces the dense-GP log-likelihood exactly (to rounding) in O(n),
//! with irregular sampling handled by the per-step `Δ`.

use crate::dual::Scalar;
use crate::linalg;
use crate::{PreparedSource, IDX_AMP, IDX_JIT, IDX_LS, IDX_WL};

const LN_2PI: f64 = 1.837_877_066_409_345_5;
const SQRT3: f64 = 1.732_050_807_568_877_2;

/// Kernel quantities derived from one hyperparameter vector.
pub(crate) struct Kernel<T> {
    /// Number of bands.
    pub b: usize,
    /// `λ = √3 / ℓ`.
    pub lam: T,
    /// Band covariance `C`, row-major `b × b`.
    pub c: Vec<T>,
    /// Extra white noise variance added to every point.
    pub jitter2: T,
}

impl<T: Scalar> Kernel<T> {
    /// Build from `theta = [ln ℓ, ln ℓ_λ, ln jitter, ln σ_0, …]` and the band
    /// wavelengths (in the same units as `ℓ_λ`).
    pub fn new(theta: &[T], wavelengths: &[f64]) -> Self {
        let b = wavelengths.len();
        let lam = (-theta[IDX_LS]).exp().scale(SQRT3);
        let half_inv_l2 = (theta[IDX_WL].scale(-2.0)).exp().scale(0.5);
        let mut c = Vec::with_capacity(b * b);
        for i in 0..b {
            for j in 0..b {
                let dl = wavelengths[i] - wavelengths[j];
                c.push((theta[IDX_AMP + i] + theta[IDX_AMP + j] - half_inv_l2.scale(dl * dl)).exp());
            }
        }
        Kernel { b, lam, c, jitter2: theta[IDX_JIT].scale(2.0).exp() }
    }

    /// Stationary state covariance `C ⊗ diag(1, λ²)`.
    fn stationary(&self) -> Vec<T> {
        let d = 2 * self.b;
        let lam2 = self.lam * self.lam;
        let mut p = vec![T::cst(0.0); d * d];
        for i in 0..self.b {
            for j in 0..self.b {
                let cij = self.c[i * self.b + j];
                p[(2 * i) * d + 2 * j] = cij;
                p[(2 * i + 1) * d + 2 * j + 1] = cij * lam2;
            }
        }
        p
    }
}

/// `A(Δ)` as `[a00, a01, a10, a11]` and `Q(Δ)` as `[q00, q01, q11]` for a
/// unit-variance Matérn-3/2 process.
#[inline]
fn matern32_step<T: Scalar>(lam: T, dt: f64) -> ([T; 4], [T; 3]) {
    let one = T::cst(1.0);
    let x = lam.scale(dt); // λΔ
    let e = (-x).exp();
    let a = [e * (one + x), e.scale(dt), -(e * lam * x), e * (one - x)];
    let e2 = e * e;
    let x2 = x * x;
    let lam2 = lam * lam;
    let q00 = one - e2 * ((one + x) * (one + x) + x2);
    let q01 = (e2 * lam * x2).scale(2.0);
    let q11 = lam2 * (one - e2 * (x2 + (one - x) * (one - x)));
    (a, [q00, q01, q11])
}

/// Kalman filter state over the stacked `2B`-dimensional model.
struct Filter<T> {
    d: usize,
    m: Vec<T>,
    p: Vec<T>,
    col: Vec<T>,
}

impl<T: Scalar> Filter<T> {
    fn new(k: &Kernel<T>) -> Self {
        let d = 2 * k.b;
        Filter { d, m: vec![T::cst(0.0); d], p: k.stationary(), col: vec![T::cst(0.0); d] }
    }

    /// Propagate by `dt ≥ 0`: `m ← (I⊗A) m`, `P ← (I⊗A) P (I⊗A)ᵀ + C⊗Q`.
    fn predict(&mut self, k: &Kernel<T>, dt: f64) {
        if dt <= 0.0 {
            return;
        }
        let (a, q) = matern32_step(k.lam, dt);
        let d = self.d;
        for i in 0..k.b {
            let (m0, m1) = (self.m[2 * i], self.m[2 * i + 1]);
            self.m[2 * i] = a[0] * m0 + a[1] * m1;
            self.m[2 * i + 1] = a[2] * m0 + a[3] * m1;
        }
        // Blocks (i, j) with j ≥ i, mirrored: A Pᵢⱼ Aᵀ + Cᵢⱼ Q.
        for i in 0..k.b {
            for j in i..k.b {
                let (r0, r1) = (2 * i, 2 * i + 1);
                let (c0, c1) = (2 * j, 2 * j + 1);
                let p0 = self.p[r0 * d + c0];
                let p1 = self.p[r0 * d + c1];
                let p2 = self.p[r1 * d + c0];
                let p3 = self.p[r1 * d + c1];
                let t0 = a[0] * p0 + a[1] * p2;
                let t1 = a[0] * p1 + a[1] * p3;
                let t2 = a[2] * p0 + a[3] * p2;
                let t3 = a[2] * p1 + a[3] * p3;
                let cij = k.c[i * k.b + j];
                let n0 = t0 * a[0] + t1 * a[1] + cij * q[0];
                let n1 = t0 * a[2] + t1 * a[3] + cij * q[1];
                let n2 = t2 * a[0] + t3 * a[1] + cij * q[1];
                let n3 = t2 * a[2] + t3 * a[3] + cij * q[2];
                self.p[r0 * d + c0] = n0;
                self.p[r0 * d + c1] = n1;
                self.p[r1 * d + c0] = n2;
                self.p[r1 * d + c1] = n3;
                if j != i {
                    self.p[c0 * d + r0] = n0;
                    self.p[c1 * d + r0] = n1;
                    self.p[c0 * d + r1] = n2;
                    self.p[c1 * d + r1] = n3;
                }
            }
        }
    }

    /// Condition on `y = f_band + ε`, `ε ~ N(0, r)`. Returns `(innovation, S)`.
    fn update(&mut self, band: usize, y: f64, r: T) -> (T, T) {
        let d = self.d;
        let idx = 2 * band;
        let s = self.p[idx * d + idx] + r;
        let v = T::cst(y) - self.m[idx];
        let inv_s = T::cst(1.0) / s;
        for j in 0..d {
            self.col[j] = self.p[j * d + idx];
        }
        let gain = v * inv_s;
        for j in 0..d {
            self.m[j] += self.col[j] * gain;
        }
        for j in 0..d {
            let cj = self.col[j] * inv_s;
            for l in j..d {
                let delta = cj * self.col[l];
                self.p[j * d + l] -= delta;
                if l != j {
                    self.p[l * d + j] -= delta;
                }
            }
        }
        (v, s)
    }
}

/// Exact GP log marginal likelihood `ln p(y | θ)` in O(n).
///
/// Generic so that calling it with [`crate::dual::Dual`] inputs yields the
/// gradient in the same pass. Returns NaN if an innovation variance is not
/// positive (only possible for pathological `θ`).
pub fn log_likelihood<T: Scalar>(prep: &PreparedSource, theta: &[T]) -> T {
    let k = Kernel::new(theta, &prep.wavelengths);
    let mut f = Filter::new(&k);
    let mut ll = T::cst(0.0);
    let mut t_prev = prep.times.first().copied().unwrap_or(0.0);
    for i in 0..prep.times.len() {
        f.predict(&k, prep.times[i] - t_prev);
        t_prev = prep.times[i];
        let r = T::cst(prep.flux_err[i] * prep.flux_err[i]) + k.jitter2;
        let (v, s) = f.update(prep.band_idx[i], prep.flux[i], r);
        if !(s.val() > 0.0) {
            return T::cst(f64::NAN);
        }
        ll -= (v * v / s + s.ln() + T::cst(LN_2PI)).scale(0.5);
    }
    ll
}

/// Forward Kalman pass over observations merged with query times, keeping
/// everything the backward passes (RTS smoothing, posterior sampling) need.
struct ForwardPass {
    d: usize,
    b: usize,
    lam: f64,
    /// Time order: `Ok(obs index)` or `Err(query index)`. At equal times
    /// observations go first, so a query sees every point at its own epoch.
    events: Vec<(f64, Result<usize, usize>)>,
    dts: Vec<f64>,
    m_pred: Vec<Vec<f64>>,
    p_pred: Vec<Vec<f64>>,
    m_filt: Vec<Vec<f64>>,
    p_filt: Vec<Vec<f64>>,
}

impl ForwardPass {
    fn run(prep: &PreparedSource, theta: &[f64], t_query: &[f64]) -> Self {
        let k = Kernel::new(theta, &prep.wavelengths);
        let mut events: Vec<(f64, Result<usize, usize>)> = Vec::with_capacity(prep.times.len() + t_query.len());
        events.extend(prep.times.iter().enumerate().map(|(i, &t)| (t, Ok(i))));
        events.extend(t_query.iter().enumerate().map(|(q, &t)| (t, Err(q))));
        events.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.is_err().cmp(&b.1.is_err()))
        });

        let n = events.len();
        let mut f = Filter::new(&k);
        let mut fp = ForwardPass {
            d: 2 * k.b,
            b: k.b,
            lam: k.lam,
            events: Vec::new(),
            dts: Vec::with_capacity(n),
            m_pred: Vec::with_capacity(n),
            p_pred: Vec::with_capacity(n),
            m_filt: Vec::with_capacity(n),
            p_filt: Vec::with_capacity(n),
        };
        let mut t_prev = events.first().map_or(0.0, |e| e.0);
        for &(t, ev) in &events {
            let dt = t - t_prev;
            t_prev = t;
            f.predict(&k, dt);
            fp.dts.push(dt);
            fp.m_pred.push(f.m.clone());
            fp.p_pred.push(f.p.clone());
            if let Ok(i) = ev {
                let r = prep.flux_err[i] * prep.flux_err[i] + k.jitter2;
                f.update(prep.band_idx[i], prep.flux[i], r);
            }
            fp.m_filt.push(f.m.clone());
            fp.p_filt.push(f.p.clone());
        }
        fp.events = events;
        fp
    }

    /// Backward gain `G = P_f[k] Aᵀ P_p[k+1]⁻¹` for `step = k`, returned with
    /// its transpose (solved as `Gᵀ = P_p⁻¹ (A P_f[k])`).
    fn gain(&self, step: usize) -> (Vec<f64>, Vec<f64>) {
        let d = self.d;
        let a = state_transition(self.lam, self.dts[step + 1], self.b);
        let mut gt = linalg::matmul(&a, &self.p_filt[step], d);
        let l = linalg::cholesky_regularised(&self.p_pred[step + 1], d)
            .expect("predicted covariance is not finite");
        let mut col = vec![0.0; d];
        for c in 0..d {
            for r in 0..d {
                col[r] = gt[r * d + c];
            }
            linalg::chol_solve(&l, d, &mut col);
            for r in 0..d {
                gt[r * d + c] = col[r];
            }
        }
        (linalg::transpose(&gt, d), gt)
    }
}

/// Posterior mean and variance of the latent `f_b(t)` for every band at
/// shifted query times (`[band][query]`), via an RTS backward pass.
///
/// Query points are merged into the time-ordered observation sequence as
/// steps with no measurement update, so the cost is O((n + m) · (2B)³).
pub fn smooth_all(prep: &PreparedSource, theta: &[f64], t_query: &[f64]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let fp = ForwardPass::run(prep, theta, t_query);
    let (d, b, n) = (fp.d, fp.b, fp.events.len());
    let mut mean = vec![vec![0.0; t_query.len()]; b];
    let mut var = vec![vec![0.0; t_query.len()]; b];
    let mut record = |ev: Result<usize, usize>, m: &[f64], p: &[f64]| {
        if let Err(q) = ev {
            for band in 0..b {
                let idx = 2 * band;
                mean[band][q] = m[idx];
                var[band][q] = p[idx * d + idx].max(0.0);
            }
        }
    };
    let mut m_s = fp.m_filt[n - 1].clone();
    let mut p_s = fp.p_filt[n - 1].clone();
    record(fp.events[n - 1].1, &m_s, &p_s);

    for step in (0..n - 1).rev() {
        let (g, gt) = fp.gain(step);
        let dm: Vec<f64> = (0..d).map(|i| m_s[i] - fp.m_pred[step + 1][i]).collect();
        let dp: Vec<f64> = (0..d * d).map(|i| p_s[i] - fp.p_pred[step + 1][i]).collect();
        m_s = (0..d)
            .map(|i| fp.m_filt[step][i] + (0..d).map(|j| g[i * d + j] * dm[j]).sum::<f64>())
            .collect();
        let g_dp_gt = linalg::matmul(&linalg::matmul(&g, &dp, d), &gt, d);
        p_s = (0..d * d).map(|i| fp.p_filt[step][i] + g_dp_gt[i]).collect();
        record(fp.events[step].1, &m_s, &p_s);
    }
    (mean, var)
}

/// [`smooth_all`] for a single band.
pub(crate) fn smooth(prep: &PreparedSource, theta: &[f64], t_query: &[f64], band: usize) -> (Vec<f64>, Vec<f64>) {
    let (mut mean, mut var) = smooth_all(prep, theta, t_query);
    (mean.swap_remove(band), var.swap_remove(band))
}

/// Joint posterior draws of every band's latent curve at shifted query times,
/// `[sample][band][query]`, by forward filtering / backward sampling:
///
///   x_N ~ N(m_f[N], P_f[N]);   x_k | x_{k+1} ~ N(m_f[k] + G_k (x_{k+1} − m_p[k+1]),
///                                               P_f[k] − G_k P_p[k+1] G_kᵀ).
///
/// Exact for the GP conditioned on `theta`, O((n + m) · (2B)²) per draw after
/// an O((n + m) · (2B)³) set-up shared by all draws.
pub fn sample_posterior(
    prep: &PreparedSource,
    theta: &[f64],
    t_query: &[f64],
    n_samples: usize,
    rng: &mut crate::rng::Rng,
) -> Vec<Vec<Vec<f64>>> {
    let fp = ForwardPass::run(prep, theta, t_query);
    let (d, b, n) = (fp.d, fp.b, fp.events.len());

    // Per step: gain and Cholesky factor of the conditional covariance.
    let steps: Vec<(Vec<f64>, Vec<f64>)> = (0..n - 1)
        .map(|step| {
            let (g, gt) = fp.gain(step);
            let gpg = linalg::matmul(&linalg::matmul(&g, &fp.p_pred[step + 1], d), &gt, d);
            let mut cov: Vec<f64> = (0..d * d).map(|i| fp.p_filt[step][i] - gpg[i]).collect();
            let scale = trace(&fp.p_filt[step], d);
            (g, psd_factor(&mut cov, d, scale))
        })
        .collect();
    let mut p_last = fp.p_filt[n - 1].clone();
    let l_last = psd_factor(&mut p_last, d, trace(&fp.p_filt[n - 1], d));

    let draw = |l: &[f64], rng: &mut crate::rng::Rng, out: &mut [f64]| {
        let z: Vec<f64> = (0..d).map(|_| rng.normal()).collect();
        for i in 0..d {
            out[i] += (0..=i).map(|k| l[i * d + k] * z[k]).sum::<f64>();
        }
    };

    let mut samples = vec![vec![vec![0.0; t_query.len()]; b]; n_samples];
    let mut x = vec![0.0; d];
    for sample in samples.iter_mut() {
        x.copy_from_slice(&fp.m_filt[n - 1]);
        draw(&l_last, rng, &mut x);
        let mut record = |ev: Result<usize, usize>, x: &[f64]| {
            if let Err(q) = ev {
                for band in 0..b {
                    sample[band][q] = x[2 * band];
                }
            }
        };
        record(fp.events[n - 1].1, &x);
        for step in (0..n - 1).rev() {
            let (g, l) = &steps[step];
            let dx: Vec<f64> = (0..d).map(|i| x[i] - fp.m_pred[step + 1][i]).collect();
            for i in 0..d {
                x[i] = fp.m_filt[step][i] + (0..d).map(|j| g[i * d + j] * dx[j]).sum::<f64>();
            }
            draw(l, rng, &mut x);
            record(fp.events[step].1, &x);
        }
    }
    samples
}

fn trace(a: &[f64], d: usize) -> f64 {
    (0..d).map(|i| a[i * d + i].abs()).sum::<f64>().max(1e-300)
}

/// Lower factor `L` with `L Lᵀ ≈ cov` for a PSD matrix that may be exactly
/// zero (simultaneous epochs) or carry rounding-level negative eigenvalues.
/// Jitter is scaled to `scale` (the typical state variance), and the factor
/// falls back to zero, i.e. a deterministic step, if nothing works.
fn psd_factor(cov: &mut [f64], d: usize, scale: f64) -> Vec<f64> {
    symmetrise(cov, d);
    let mut eps = 1e-13 * scale;
    while eps <= 1e-6 * scale {
        let mut l = cov.to_vec();
        for i in 0..d {
            l[i * d + i] += eps;
        }
        if linalg::cholesky(&mut l, d) {
            return l;
        }
        eps *= 10.0;
    }
    vec![0.0; d * d]
}

fn symmetrise(a: &mut [f64], d: usize) {
    for i in 0..d {
        for j in 0..i {
            let s = 0.5 * (a[i * d + j] + a[j * d + i]);
            a[i * d + j] = s;
            a[j * d + i] = s;
        }
    }
}

/// Dense `I_B ⊗ A(Δ)` (f64), for the smoother.
fn state_transition(lam: f64, dt: f64, b: usize) -> Vec<f64> {
    let d = 2 * b;
    let mut a = vec![0.0; d * d];
    let blk = if dt > 0.0 { matern32_step(lam, dt).0 } else { [1.0, 0.0, 0.0, 1.0] };
    for i in 0..b {
        a[(2 * i) * d + 2 * i] = blk[0];
        a[(2 * i) * d + 2 * i + 1] = blk[1];
        a[(2 * i + 1) * d + 2 * i] = blk[2];
        a[(2 * i + 1) * d + 2 * i + 1] = blk[3];
    }
    a
}

/// Process-noise and stationary covariances for simulation: returns
/// (`I⊗A(Δ)`, `C⊗Q(Δ)`) or, for `dt = None`, (`I`, `C⊗P∞`).
pub(crate) fn transition_and_noise(theta: &[f64], wavelengths: &[f64], dt: Option<f64>) -> (Vec<f64>, Vec<f64>) {
    let k = Kernel::new(theta, wavelengths);
    let d = 2 * k.b;
    match dt {
        None => {
            let mut eye = vec![0.0; d * d];
            for i in 0..d {
                eye[i * d + i] = 1.0;
            }
            (eye, k.stationary())
        }
        Some(dt) => {
            let (_, q) = matern32_step(k.lam, dt);
            let mut qf = vec![0.0; d * d];
            for i in 0..k.b {
                for j in 0..k.b {
                    let cij = k.c[i * k.b + j];
                    qf[(2 * i) * d + 2 * j] = cij * q[0];
                    qf[(2 * i) * d + 2 * j + 1] = cij * q[1];
                    qf[(2 * i + 1) * d + 2 * j] = cij * q[1];
                    qf[(2 * i + 1) * d + 2 * j + 1] = cij * q[2];
                }
            }
            (state_transition(k.lam, dt, k.b), qf)
        }
    }
}
