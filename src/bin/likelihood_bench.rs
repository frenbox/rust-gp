//! Single-core timings on synthetic two-band light curves:
//! dense vs Kalman likelihood, gradient cost, full fit, and warm refit.
//!
//! Usage: likelihood-bench

use std::hint::black_box;
use std::time::Instant;

use rust_gp::synthetic::{simulate, SimSpec};
use rust_gp::*;

/// Mean seconds per call, repeating for at least `min_secs`.
fn time<F: FnMut()>(mut f: F, min_secs: f64) -> f64 {
    f();
    let mut reps = 1usize;
    loop {
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        let el = t.elapsed().as_secs_f64();
        if el >= min_secs {
            return el / reps as f64;
        }
        reps = (reps * 2).max((reps as f64 * min_secs / el.max(1e-9)) as usize);
    }
}

fn fmt(s: f64) -> String {
    if s < 1e-3 {
        format!("{:.1} µs", s * 1e6)
    } else if s < 1.0 {
        format!("{:.2} ms", s * 1e3)
    } else {
        format!("{s:.2} s")
    }
}

fn main() {
    let config = GpConfig::default();
    println!(
        "{:>6}  {:>11}  {:>11}  {:>10}  {:>11}  {:>11}  {:>9}  {:>11}  {:>6}",
        "n", "dense ll", "kalman ll", "|Δll|", "ll+grad", "fit (3x)", "fits/s", "warm +1pt", "agree"
    );
    println!("{}", "-".repeat(105));
    for &n in &[30usize, 100, 300, 1000, 3000, 10000] {
        let spec = SimSpec::two_band(n, 60.0 * (n as f64 / 100.0).max(1.0));
        let obs = simulate(&spec, n as u64);
        let prep = prepare(&obs, &config).unwrap();
        let theta = prep.theta_from_hyper(&spec.hyper);

        let kll = kalman::log_likelihood(&prep, &theta);
        let (t_dense, diff) = if n <= 1000 {
            let dll = dense::log_likelihood(&prep, &theta);
            (Some(time(|| { black_box(dense::log_likelihood(&prep, black_box(&theta))); }, 0.3)), Some((kll - dll).abs()))
        } else {
            (None, None)
        };
        let t_k = time(|| { black_box(kalman::log_likelihood(&prep, black_box(&theta))); }, 0.3);
        let mut g = vec![0.0; theta.len()];
        let t_g = time(|| { black_box(neg_log_posterior_grad(&prep, black_box(&theta), &mut g)); }, 0.3);

        let mut res = None;
        let t_fit = time(|| res = Some(fit_prepared(&prep, &config)), 0.5);
        let res = res.unwrap();

        // Streaming update: fit n−1 points, then add the last one warm.
        let prev = fit(&obs[..n - 1], &config).unwrap();
        let t_warm = time(|| { black_box(fit_prepared_warm(&prep, &config, &prev.hyper)); }, 0.5);

        println!(
            "{:>6}  {:>11}  {:>11}  {:>10}  {:>11}  {:>11}  {:>9.0}  {:>11}  {:>4}/{}",
            n,
            t_dense.map_or("—".into(), fmt),
            fmt(t_k),
            diff.map_or("—".into(), |d| format!("{d:.1e}")),
            fmt(t_g),
            fmt(t_fit),
            1.0 / t_fit,
            fmt(t_warm),
            res.n_starts_agree,
            res.n_starts,
        );
    }
}
