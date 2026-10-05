//! Feature-extraction tests: no data files required.
//!
//!   cargo test --release --test test_features -- --nocapture

use rust_gp::features::{light_curve_features, measure, FeatureConfig, HALF_MAX_MAG};
use rust_gp::rng::Rng;
use rust_gp::synthetic::{simulate, SimSpec};
use rust_gp::*;

/// Bazin profile: exponential rise and decline.
fn bazin(t: f64, t0: f64, tau_rise: f64, tau_fall: f64) -> f64 {
    (-(t - t0) / tau_fall).exp() / (1.0 + (-(t - t0) / tau_rise).exp())
}

/// Two-band Bazin transient sampled every `cadence` days over `[t_start, t_end]`
/// with 2 % noise; r is brighter and declines more slowly than g.
fn bazin_obs(t_start: f64, t_end: f64, cadence: f64, seed: u64) -> Vec<Obs> {
    let mut rng = Rng::new(seed);
    let mut obs = Vec::new();
    let mut t = t_start;
    while t <= t_end {
        for (band, amp, fall) in [("g", 800.0, 12.0), ("r", 1000.0, 20.0)] {
            let f = amp * bazin(t, 0.0, 3.0, fall);
            let err = 0.02 * amp;
            obs.push(Obs { time: 60000.0 + t, flux: f + err * rng.normal(), flux_err: err, band: band.into(), upper_limit: false });
        }
        t += cadence;
    }
    obs
}

/// True features of the noiseless Bazin curve, measured on a fine grid.
fn bazin_truth(amp: f64, fall: f64) -> (f64, f64, f64, f64) {
    let t: Vec<f64> = (0..200_001).map(|i| -40.0 + 160.0 * i as f64 / 200_000.0).collect();
    let f: Vec<f64> = t.iter().map(|&x| amp * bazin(x, 0.0, 3.0, fall)).collect();
    let s = measure(&t, &f).unwrap();
    (s.peak_time + 60000.0, 23.9 - 2.5 * s.peak_flux.log10(), s.t_rise.unwrap(), s.t_fade.unwrap())
}

#[test]
fn measure_is_exact_on_a_gaussian() {
    let sigma = 4.0;
    let t: Vec<f64> = (0..2001).map(|i| i as f64 * 0.05).collect();
    let f: Vec<f64> = t.iter().map(|&x| (-(x - 50.0f64).powi(2) / (2.0 * sigma * sigma)).exp()).collect();
    let s = measure(&t, &f).unwrap();
    let half_width = sigma * (2.0 * 2f64.ln()).sqrt();
    assert!((s.peak_time - 50.0).abs() < 1e-6);
    assert!((s.peak_flux - 1.0).abs() < 1e-6);
    assert!((s.t_rise.unwrap() - half_width).abs() < 1e-3);
    assert!((s.t_fade.unwrap() - half_width).abs() < 1e-3);
    assert!(s.interior);
}

#[test]
fn posterior_draws_match_smoother_moments() {
    let obs = simulate(&SimSpec::two_band(40, 60.0), 21);
    let prep = prepare(&obs, &GpConfig::default()).unwrap();
    let r = fit_prepared(&prep, &GpConfig::default());
    let theta = prep.theta_from_hyper(&r.hyper);
    let grid: Vec<f64> = (0..30).map(|i| -5.0 + 2.5 * i as f64).collect();
    let (mean, var) = kalman::smooth_all(&prep, &theta, &grid);
    let n = 4000;
    let samples = kalman::sample_posterior(&prep, &theta, &grid, n, &mut Rng::new(3));
    let mut worst_z: f64 = 0.0;
    let mut worst_var: f64 = 0.0;
    for b in 0..prep.n_bands() {
        for q in 0..grid.len() {
            let xs: Vec<f64> = samples.iter().map(|s| s[b][q]).collect();
            let m = xs.iter().sum::<f64>() / n as f64;
            let v = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1) as f64;
            worst_z = worst_z.max((m - mean[b][q]).abs() / (var[b][q] / n as f64).sqrt());
            worst_var = worst_var.max((v / var[b][q] - 1.0).abs());
        }
    }
    println!("worst mean z = {worst_z:.2}, worst relative variance error = {worst_var:.3}");
    assert!(worst_z < 4.5, "sample means disagree with the smoother");
    assert!(worst_var < 0.15, "sample variances disagree with the smoother");
}

#[test]
fn recovers_bazin_features() {
    let obs = bazin_obs(-15.0, 90.0, 1.5, 7);
    let prep = prepare(&obs, &GpConfig::default()).unwrap();
    let r = fit_prepared(&prep, &GpConfig::default());
    let feats = light_curve_features(&prep, &r.hyper, &FeatureConfig::default());
    for (f, (amp, fall)) in feats.iter().zip([(800.0, 12.0), (1000.0, 20.0)]) {
        let (tp, mp, tr, tf) = bazin_truth(amp, fall);
        println!(
            "{}: peak {:.2}±{:.2} (true {tp:.2})  mag {:.3}±{:.3} (true {mp:.3})  \
             t_rise {:.2}±{:.2} (true {tr:.2})  t_fade {:.2}±{:.2} (true {tf:.2})  fwhm {:.2}±{:.2}",
            f.band,
            f.peak_time.unwrap(), f.peak_time_err.unwrap(),
            f.peak_mag.unwrap(), f.peak_mag_err.unwrap(),
            f.t_rise.unwrap(), f.t_rise_err.unwrap(),
            f.t_fade.unwrap(), f.t_fade_err.unwrap(),
            f.fwhm.unwrap(), f.fwhm_err.unwrap(),
        );
        assert!(f.peak_bracketed);
        assert!((f.peak_mag.unwrap() - mp).abs() < 0.03, "{} peak mag", f.band);
        assert!((f.t_rise.unwrap() / tr - 1.0).abs() < 0.10, "{} t_rise", f.band);
        assert!((f.t_fade.unwrap() / tf - 1.0).abs() < 0.05, "{} t_fade", f.band);
        assert!((f.rise_rate.unwrap() - HALF_MAX_MAG / f.t_rise.unwrap()).abs() < 1e-12);
        assert!((f.fwhm.unwrap() - f.t_rise.unwrap() - f.t_fade.unwrap()).abs() < 1e-12);
    }
}

#[test]
fn unobserved_rise_or_fade_is_none() {
    let config = GpConfig::default();
    let fcfg = FeatureConfig { n_samples: 0, ..FeatureConfig::default() };
    // Starts after the peak: the peak is the first detection, so nothing is
    // reported (a "fade" from the first point would not be a fade time).
    let late = bazin_obs(8.0, 90.0, 1.5, 1);
    let prep = prepare(&late, &config).unwrap();
    let r = fit_prepared(&prep, &config);
    for f in light_curve_features(&prep, &r.hyper, &fcfg) {
        assert!(!f.peak_bracketed);
        assert!(f.peak_mag.is_none() && f.t_rise.is_none() && f.t_fade.is_none(), "{f:?}");
    }
    // Starts just before the peak: peak bracketed but rise not seen to half-max.
    let near = bazin_obs(1.0, 90.0, 1.0, 3);
    let prep = prepare(&near, &config).unwrap();
    let r = fit_prepared(&prep, &config);
    for f in light_curve_features(&prep, &r.hyper, &fcfg) {
        assert!(f.peak_bracketed, "{f:?}");
        assert!(f.t_rise.is_none() && f.rise_rate.is_none() && f.fwhm.is_none(), "{f:?}");
        assert!(f.t_fade.is_some());
    }
    // Ends after the peak but before half-max: no fade.
    let early = bazin_obs(-15.0, 9.0, 1.0, 2);
    let prep = prepare(&early, &config).unwrap();
    let r = fit_prepared(&prep, &config);
    for f in light_curve_features(&prep, &r.hyper, &fcfg) {
        assert!(f.t_fade.is_none() && f.fade_rate.is_none() && f.fwhm.is_none(), "{f:?}");
        assert!(f.t_rise.is_some());
    }
}

#[test]
fn sparse_band_gets_no_features() {
    let mut obs = bazin_obs(-15.0, 90.0, 3.0, 4);
    obs.push(Obs { time: 60010.0, flux: 500.0, flux_err: 20.0, band: "i".into(), upper_limit: false });
    let prep = prepare(&obs, &GpConfig::default()).unwrap();
    let r = fit_prepared(&prep, &GpConfig::default());
    let feats = light_curve_features(&prep, &r.hyper, &FeatureConfig { n_samples: 0, ..Default::default() });
    let i = feats.iter().find(|f| f.band == "i").unwrap();
    assert_eq!(i.n_points, 1);
    assert!(i.peak_mag.is_none());
}

#[test]
fn non_detections_recover_an_unobserved_rise() {
    // Detections start at day +1, already above half-max (crossing ≈ −2.8 d).
    let mut obs = bazin_obs(1.0, 90.0, 1.5, 5);
    let config = GpConfig::default();
    let fcfg = FeatureConfig::default();
    let prep = prepare(&obs, &config).unwrap();
    let before = light_curve_features(&prep, &fit_prepared(&prep, &config).hyper, &fcfg);
    assert!(before.iter().all(|f| f.t_rise.is_none()));

    // Pre-discovery 5σ limits of 250 µJy every 3 days, as load_csv_with_limits
    // would enter them: flux 0 ± f_lim/5. (At day −6 the true r flux, ~160 µJy,
    // is below the limit but 3σ from zero: the approximation's known bias.)
    for t in [-15.0, -12.0, -9.0, -6.0] {
        for band in ["g", "r"] {
            obs.push(Obs { time: 60000.0 + t, flux: 0.0, flux_err: 50.0, band: band.into(), upper_limit: true });
        }
    }
    let prep = prepare(&obs, &config).unwrap();
    let r = fit_prepared(&prep, &config);
    assert_eq!((r.n_obs, r.n_limits), (prep.n_detections(), 8));
    let after = light_curve_features(&prep, &r.hyper, &fcfg);
    for (f, (amp, fall)) in after.iter().zip([(800.0, 12.0), (1000.0, 20.0)]) {
        let (_, _, tr, _) = bazin_truth(amp, fall);
        let (est, err) = (f.t_rise.expect("rise should now be bracketed"), f.t_rise_err.unwrap());
        println!("{}: t_rise {est:.2} ± {err:.2} (true {tr:.2}), n_points {}", f.band, f.n_points);
        assert!((est - tr).abs() < 3.0 * err + 0.5, "{}: {est} vs {tr}", f.band);
        assert_eq!(f.n_points, prep.band_idx.iter().zip(&prep.upper_limit).filter(|(&b, &l)| prep.bands[b] == f.band && !l).count());
    }
}
