//! Minimal dense linear algebra on row-major `n × n` slices (f64 only).
//!
//! Used by the RTS smoother (state dimension ≤ 12), the Laplace covariance
//! (≤ 9 parameters) and the dense reference GP in [`crate::dense`].

/// In-place lower Cholesky factor of a symmetric positive-definite matrix.
/// The strict upper triangle is zeroed. Returns `false` if not PD.
pub fn cholesky(a: &mut [f64], n: usize) -> bool {
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= a[j * n + k] * a[j * n + k];
        }
        if !(d > 0.0) || !d.is_finite() {
            return false;
        }
        let d = d.sqrt();
        a[j * n + j] = d;
        for i in (j + 1)..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = s / d;
        }
        for k in (j + 1)..n {
            a[j * n + k] = 0.0;
        }
    }
    true
}

/// Cholesky with escalating diagonal regularisation for matrices that are PSD
/// but numerically singular. Returns the factor, or `None` if even heavy
/// regularisation fails (non-finite input).
pub fn cholesky_regularised(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let trace: f64 = (0..n).map(|i| a[i * n + i].abs()).sum::<f64>().max(1e-300);
    let mut eps = 0.0;
    for _ in 0..12 {
        let mut l = a.to_vec();
        for i in 0..n {
            l[i * n + i] += eps;
        }
        if cholesky(&mut l, n) {
            return Some(l);
        }
        eps = if eps == 0.0 { 1e-14 * trace } else { eps * 10.0 };
    }
    None
}

/// Solve `L Lᵀ x = b` in place given the lower Cholesky factor `l`.
pub fn chol_solve(l: &[f64], n: usize, b: &mut [f64]) {
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * b[k];
        }
        b[i] = s / l[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in (i + 1)..n {
            s -= l[k * n + i] * b[k];
        }
        b[i] = s / l[i * n + i];
    }
}

/// `C = A · B` for row-major `n × n` matrices.
pub fn matmul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i * n + k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..n {
                c[i * n + j] += aik * b[k * n + j];
            }
        }
    }
    c
}

/// Transpose of a row-major `n × n` matrix.
pub fn transpose(a: &[f64], n: usize) -> Vec<f64> {
    let mut t = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            t[j * n + i] = a[i * n + j];
        }
    }
    t
}
