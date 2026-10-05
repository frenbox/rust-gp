//! Blackbody temperature / cooling-rate tests: no data files required.
//!
//!   cargo test --release --test test_thermal -- --nocapture

use rust_gp::rng::Rng;
use rust_gp::thermal::{fit_blackbody, thermal_evolution, ThermalConfig};
use rust_gp::*;

const HC_OVER_K: f64 = 1.438_776_877e8;

fn bnu(lambda_aa: f64, t: f64) -> f64 {
    lambda_aa.powi(-3) / (HC_OVER_K / (lambda_aa * t)).exp_m1()
}

#[test]
fn blackbody_fit_is_exact_on_noiseless_seds() {
    let lam = [4.770, 6.231, 7.625];
    for t in [5000.0, 8000.0, 12000.0, 20000.0] {
        let f: Vec<f64> = lam.iter().map(|&l| 3.0 * bnu(l * 1000.0, t) / bnu(6231.0, t)).collect();
        let (log_t, _, chi2) = fit_blackbody(&lam, &f, &[1e-4; 3]).unwrap();
        println!("T={t}: fitted {:.1} K, chi2 {chi2:.1e}", 10f64.powf(log_t));
        assert!((log_t - t.log10()).abs() < 1e-6);
    }
    // Out-of-range temperatures are refused rather than pinned to the bound.
    let f: Vec<f64> = lam.iter().map(|&l| bnu(l * 1000.0, 300_000.0)).collect();
    assert!(fit_blackbody(&lam, &f, &[1e-4; 3]).is_none());
}

/// g/r/i transient: Bazin envelope in r, blackbody colours with
/// log10 T(t) = 4.05 − 0.006 (t − 5) d, 2 % noise, 1.5 d cadence.
fn cooling_transient(seed: u64) -> (Vec<Obs>, impl Fn(f64) -> f64) {
    let log_t = |t: f64| 4.05 - 0.006 * (t - 5.0);
    let env = |t: f64| 1000.0 * (-t / 20.0).exp() / (1.0 + (-t / 3.0).exp());
    let mut rng = Rng::new(seed);
    let mut obs = Vec::new();
    let mut t = -15.0;
    while t <= 70.0 {
        let temp = 10f64.powf(log_t(t));
        for (band, lam) in [("g", 4770.0), ("r", 6231.0), ("i", 7625.0)] {
            let f = env(t) * bnu(lam, temp) / bnu(6231.0, temp);
            let err = 0.02 * env(t).max(50.0);
            obs.push(Obs { time: 60000.0 + t, flux: f + err * rng.normal(), flux_err: err, band: band.into(), upper_limit: false });
        }
        t += 1.5;
    }
    (obs, log_t)
}

#[test]
fn recovers_peak_temperature_and_cooling_rate() {
    let config = GpConfig::default();
    for seed in [1, 2, 3] {
        let (obs, log_t) = cooling_transient(seed);
        let prep = prepare(&obs, &config).unwrap();
        let r = fit_prepared(&prep, &config);
        let th = thermal_evolution(&prep, &r.hyper, &ThermalConfig::default());
        let tp = th.peak_time.unwrap() - 60000.0;
        let (lp, lp_err) = (th.log_temp_peak.unwrap(), th.log_temp_peak_err.unwrap());
        let (cr, cr_err) = (th.cooling_rate.unwrap(), th.cooling_rate_err.unwrap());
        println!(
            "seed {seed}: ref {:?} peak t={tp:.2}  log T_peak {lp:.3}±{lp_err:.3} (true {:.3})  \
             cooling {cr:+.4}±{cr_err:.4} dex/d (true −0.0060)  {} epochs",
            th.ref_band, log_t(tp), th.times.len()
        );
        assert!((lp - log_t(tp)).abs() < 3.0 * lp_err + 0.02, "seed {seed}: T_peak");
        assert!((cr + 0.006).abs() < 3.0 * cr_err + 0.001, "seed {seed}: cooling");
        assert!(th.log_temp_err.iter().all(|e| e.is_some()));
    }
}

#[test]
fn single_band_gives_no_temperature() {
    let (obs, _) = cooling_transient(4);
    let r_only: Vec<Obs> = obs.into_iter().filter(|o| o.band == "r").collect();
    let prep = prepare(&r_only, &GpConfig::default()).unwrap();
    let r = fit_prepared(&prep, &GpConfig::default());
    let th = thermal_evolution(&prep, &r.hyper, &ThermalConfig::default());
    assert!(th.times.is_empty() && th.log_temp_peak.is_none() && th.cooling_rate.is_none());
}
