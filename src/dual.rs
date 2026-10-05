//! Forward-mode dual numbers.
//!
//! The likelihood is written once, generic over [`Scalar`], and evaluated
//! either with `f64` (value only) or with [`Dual<N>`] (value plus the exact
//! gradient with respect to all `N` hyperparameters in the same pass). With
//! at most nine hyperparameters this costs roughly `N + 1` times a plain
//! evaluation, and needs no hand-derived gradient of the Kalman recursion.

use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

/// The arithmetic the likelihood needs: field operations plus `exp`, `ln`.
pub trait Scalar:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
{
    /// A constant (zero derivative).
    fn cst(v: f64) -> Self;
    /// The value part.
    fn val(self) -> f64;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    /// Multiply by a plain `f64` without promoting it to `Self` first.
    fn scale(self, k: f64) -> Self;
}

impl Scalar for f64 {
    #[inline]
    fn cst(v: f64) -> Self {
        v
    }
    #[inline]
    fn val(self) -> f64 {
        self
    }
    #[inline]
    fn exp(self) -> Self {
        f64::exp(self)
    }
    #[inline]
    fn ln(self) -> Self {
        f64::ln(self)
    }
    #[inline]
    fn scale(self, k: f64) -> Self {
        self * k
    }
}

/// A value together with its derivatives with respect to `N` inputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dual<const N: usize> {
    pub v: f64,
    pub d: [f64; N],
}

impl<const N: usize> Dual<N> {
    /// The `i`-th independent variable, seeded with `d = e_i`.
    pub fn var(v: f64, i: usize) -> Self {
        let mut d = [0.0; N];
        d[i] = 1.0;
        Dual { v, d }
    }

    #[inline]
    fn chain(self, v: f64, dv: f64) -> Self {
        let mut d = self.d;
        for x in &mut d {
            *x *= dv;
        }
        Dual { v, d }
    }
}

impl<const N: usize> Add for Dual<N> {
    type Output = Self;
    #[inline]
    fn add(mut self, o: Self) -> Self {
        self += o;
        self
    }
}

impl<const N: usize> AddAssign for Dual<N> {
    #[inline]
    fn add_assign(&mut self, o: Self) {
        self.v += o.v;
        for i in 0..N {
            self.d[i] += o.d[i];
        }
    }
}

impl<const N: usize> Sub for Dual<N> {
    type Output = Self;
    #[inline]
    fn sub(mut self, o: Self) -> Self {
        self -= o;
        self
    }
}

impl<const N: usize> SubAssign for Dual<N> {
    #[inline]
    fn sub_assign(&mut self, o: Self) {
        self.v -= o.v;
        for i in 0..N {
            self.d[i] -= o.d[i];
        }
    }
}

impl<const N: usize> Mul for Dual<N> {
    type Output = Self;
    #[inline]
    fn mul(self, o: Self) -> Self {
        let mut d = [0.0; N];
        for i in 0..N {
            d[i] = self.d[i] * o.v + self.v * o.d[i];
        }
        Dual { v: self.v * o.v, d }
    }
}

impl<const N: usize> Div for Dual<N> {
    type Output = Self;
    #[inline]
    fn div(self, o: Self) -> Self {
        let inv = 1.0 / o.v;
        let v = self.v * inv;
        let mut d = [0.0; N];
        for i in 0..N {
            d[i] = (self.d[i] - v * o.d[i]) * inv;
        }
        Dual { v, d }
    }
}

impl<const N: usize> Neg for Dual<N> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self.scale(-1.0)
    }
}

impl<const N: usize> Scalar for Dual<N> {
    #[inline]
    fn cst(v: f64) -> Self {
        Dual { v, d: [0.0; N] }
    }
    #[inline]
    fn val(self) -> f64 {
        self.v
    }
    #[inline]
    fn exp(self) -> Self {
        let e = self.v.exp();
        self.chain(e, e)
    }
    #[inline]
    fn ln(self) -> Self {
        self.chain(self.v.ln(), 1.0 / self.v)
    }
    #[inline]
    fn scale(self, k: f64) -> Self {
        self.chain(self.v * k, k)
    }
}
