//! Correctness tests: no data files required.
//!
//!   cargo test --release --test test_gp -- --nocapture

use rust_gp::synthetic::{simulate, SimSpec};
use rust_gp::*;

fn spec(n: usize, bands: &[&str]) -> SimSpec {
    let mut s = SimSpec::two_band(n, 60.0);
    s.hyper.amplitudes = bands.iter().enumerate().map(|(i, b)| (b.to_string(), 1.0 - 0.15 * i as f64)).collect();
    s
}

fn theta_for(prep: &PreparedSource) -> Vec<f64> {
    let mut t = vec![5f64.ln(), 3f64.ln(), 0.02f64.ln()];
    t.extend((0..prep.n_bands()).map(|b| (0.6 + 0.1 * b as f64).ln()));
    t
}

#[test]
fn kalman_likelihood_matches_dense() {
    for (bands, n) in [(&["r"][..], 40), (&["g", "r"][..], 80), (&["g", "r", "i"][..], 120)] {
        let obs = simulate(&spec(n, bands), 7);
        let prep = prepare(&obs, &GpConfig::default()).unwrap();
        let theta = theta_for(&prep);
        let k = kalman::log_likelihood(&prep, &theta);
        let d = dense::log_likelihood(&prep, &theta);
        println!("bands={} n={n}: kalman={k:.10} dense={d:.10} diff={:.2e}", bands.len(), (k - d).abs());
        assert!((k - d).abs() < 1e-8 * d.abs().max(1.0), "kalman {k} vs dense {d}");
    }
}

#[test]
fn simultaneous_epochs_in_different_bands() {
    // Same time in two bands (Δ = 0 steps) and a repeated epoch in one band.
    let mut obs = simulate(&spec(30, &["g", "r"]), 3);
    let dup: Vec<Obs> = obs.iter().take(10).map(|o| Obs { band: if o.band == "g" { "r".into() } else { "g".into() }, ..o.clone() }).collect();
    obs.extend(dup);
    obs.push(obs[0].clone());
    let prep = prepare(&obs, &GpConfig::default()).unwrap();
    let theta = theta_for(&prep);
    let k = kalman::log_likelihood(&prep, &theta);
    let d = dense::log_likelihood(&prep, &theta);
    assert!((k - d).abs() < 1e-8 * d.abs(), "kalman {k} vs dense {d}");
}

#[test]
fn dual_gradient_matches_finite_differences() {
    let obs = simulate(&spec(100, &["g", "r", "i"]), 11);
    let prep = prepare(&obs, &GpConfig::default()).unwrap();
    let theta = theta_for(&prep);
    let mut grad = vec![0.0; theta.len()];
    let f0 = neg_log_posterior_grad(&prep, &theta, &mut grad);
    let f = |t: &[f64]| -log_posterior::<f64>(&prep, t);
    assert!((f0 - f(&theta)).abs() < 1e-10 * f0.abs().max(1.0));
    for i in 0..theta.len() {
        let h = 1e-6;
        let (mut tp, mut tm) = (theta.clone(), theta.clone());
        tp[i] += h;
        tm[i] -= h;
        let fd = (f(&tp) - f(&tm)) / (2.0 * h);
        println!("θ[{i}]: dual={:+.8e} fd={fd:+.8e}", grad[i]);
        assert!((grad[i] - fd).abs() < 1e-5 * fd.abs().max(1.0), "θ[{i}]: {} vs {fd}", grad[i]);
    }
}

#[test]
fn smoother_matches_dense_posterior() {
    let obs = simulate(&spec(60, &["g", "r"]), 5);
    let prep = prepare(&obs, &GpConfig::default()).unwrap();
    let theta = theta_for(&prep);
    let hyper = prep.hyper_from_theta(&theta);
    // Query inside, between, on top of, and outside the data.
    let mut tq: Vec<f64> = (0..50).map(|i| -10.0 + 1.6 * i as f64).collect();
    tq.push(prep.times[7]);
    for (b, band) in prep.bands.iter().enumerate() {
        let (dm, dv) = dense::predict(&prep, &theta, &tq, b);
        let traw: Vec<f64> = tq.iter().map(|t| t + prep.t_ref).collect();
        let p = predict(&prep, &hyper, &traw, band, false).unwrap();
        for i in 0..tq.len() {
            let km = p.mean[i] / prep.flux_scale;
            let kv = (p.std[i] / prep.flux_scale).powi(2);
            assert!((km - dm[i]).abs() < 1e-7, "{band} t={}: mean {km} vs {}", tq[i], dm[i]);
            assert!((kv - dv[i]).abs() < 1e-7, "{band} t={}: var {kv} vs {}", tq[i], dv[i]);
        }
    }
}

#[test]
fn recovers_hyperparameters() {
    let mut log_ratio = Vec::new();
    for seed in 0..10 {
        let s = spec(300, &["g", "r"]);
        let obs = simulate(&s, 100 + seed);
        let r = fit(&obs, &GpConfig::default()).unwrap();
        assert!(r.converged, "seed {seed} did not converge");
        let lr = (r.hyper.length_scale / s.hyper.length_scale).ln();
        let err = r.ln_err.as_ref().expect("Hessian should be PD")[0];
        println!(
            "seed {seed}: ℓ={:.2} (true {}), ln ratio {lr:+.3} ± {err:.3}, agree {}/{}",
            r.hyper.length_scale, s.hyper.length_scale, r.n_starts_agree, r.n_starts
        );
        assert!(lr.abs() < 4.0 * err + 0.05, "seed {seed}: ℓ off by {lr} with σ {err}");
        log_ratio.push(lr);
    }
    let mean = log_ratio.iter().sum::<f64>() / log_ratio.len() as f64;
    assert!(mean.abs() < 0.15, "mean ln(ℓ_fit/ℓ_true) = {mean}");
}

#[test]
fn starts_agree_on_synthetic_data() {
    let mut agree = 0;
    for seed in 0..20 {
        let r = fit(&simulate(&spec(100, &["g", "r"]), 500 + seed), &GpConfig::default()).unwrap();
        agree += (r.n_starts_agree == r.n_starts) as usize;
    }
    println!("all starts agreed on {agree}/20");
    assert!(agree >= 18);
}

#[test]
fn warm_refit_matches_cold_refit() {
    let s = spec(101, &["g", "r"]);
    let obs = simulate(&s, 42);
    let config = GpConfig::default();
    let before = fit(&obs[..100], &config).unwrap();
    let cold = fit(&obs, &config).unwrap();
    let warm = fit_warm(&obs, &config, &before.hyper).unwrap();
    println!(
        "cold lp={:.8} ({} evals), warm lp={:.8} ({} evals)",
        cold.log_posterior, cold.n_evals, warm.log_posterior, warm.n_evals
    );
    assert!((cold.log_posterior - warm.log_posterior).abs() < 1e-4);
    assert!(warm.n_evals < cold.n_evals);
}

#[test]
fn rejects_too_few_points() {
    let obs = simulate(&spec(4, &["g", "r"]), 1);
    assert!(fit(&obs, &GpConfig::default()).is_err());
}

#[test]
fn single_band_and_unknown_bands() {
    let mut obs = simulate(&spec(50, &["r"]), 9);
    obs.push(Obs { time: obs[0].time, flux: 1.0, flux_err: 0.1, band: "w".into(), upper_limit: false });
    let r = fit(&obs, &GpConfig::default()).unwrap();
    assert_eq!(r.n_bands, 1);
    assert_eq!(r.n_obs, 50);
    assert!(r.converged);
}
