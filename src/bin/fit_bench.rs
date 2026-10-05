//! Fit every CSV in a directory at several thread counts and report
//! throughput, convergence and multi-start agreement.
//!
//! CSVs are loaded and prepared once up front, so the timings are pure
//! fitting. The single-thread pass also breaks timing down by light-curve
//! length.
//!
//! Usage: fit-bench [data_dir] [--threads 1,2,4,8] [--n-starts 3]

use std::time::Instant;

use rust_gp::*;
use rayon::prelude::*;

fn main() {
    let mut data_dir = "../lc_fitting_comparison/data/photometry".to_string();
    let mut threads: Vec<usize> = vec![1, 2, 4, std::thread::available_parallelism().map_or(8, |n| n.get())];
    let mut config = GpConfig::default();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--threads" => {
                threads = args.next().expect("--threads N,N,...").split(',').map(|s| s.parse().unwrap()).collect()
            }
            "--n-starts" => config.n_starts = args.next().expect("--n-starts N").parse().unwrap(),
            _ => data_dir = a,
        }
    }
    threads.dedup();

    let paths = find_csv_files(&data_dir);
    let t_load = Instant::now();
    let loaded: Vec<(String, Result<PreparedSource, String>)> = paths
        .par_iter()
        .map(|p| (p.clone(), load_csv_for(p, &config).and_then(|o| prepare(&o, &config))))
        .collect();
    let load_s = t_load.elapsed().as_secs_f64();
    let (preps, skipped): (Vec<_>, Vec<_>) = loaded.into_iter().partition(|(_, r)| r.is_ok());
    let preps: Vec<(String, PreparedSource)> = preps.into_iter().map(|(p, r)| (p, r.unwrap())).collect();
    let n_points: usize = preps.iter().map(|(_, p)| p.n_obs()).sum();
    eprintln!(
        "{} CSVs in {data_dir}: {} fittable ({} points), {} skipped; load+prepare {:.2} s",
        paths.len(),
        preps.len(),
        n_points,
        skipped.len(),
        load_s
    );
    if preps.is_empty() {
        return;
    }

    println!(
        "\n{:>8}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "threads", "total s", "ms/source", "fits/s", "speedup", "converged"
    );
    println!("{}", "-".repeat(66));
    let mut base = 0.0;
    let mut single: Vec<(usize, f64, GpResult)> = Vec::new();
    for &nt in &threads {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(nt).build().unwrap();
        let t0 = Instant::now();
        let results: Vec<(usize, f64, GpResult)> = pool.install(|| {
            preps
                .par_iter()
                .map(|(_, p)| {
                    let t = Instant::now();
                    let r = fit_prepared(p, &config);
                    (p.n_obs(), t.elapsed().as_secs_f64(), r)
                })
                .collect()
        });
        let total = t0.elapsed().as_secs_f64();
        if base == 0.0 {
            base = total;
        }
        let conv = results.iter().filter(|r| r.2.converged).count();
        println!(
            "{:>8}  {:>10.2}  {:>10.3}  {:>10.0}  {:>9.2}x  {:>6}/{}",
            nt,
            total,
            1e3 * total / preps.len() as f64,
            preps.len() as f64 / total,
            base / total,
            conv,
            results.len()
        );
        if single.is_empty() {
            single = results;
        }
    }

    // Per-size breakdown from the first (lowest thread count) pass.
    println!("\nPer-fit time by light-curve length ({}-thread pass):", threads[0]);
    println!("{:>14}  {:>8}  {:>12}  {:>12}  {:>14}", "n_obs", "sources", "median ms", "max ms", "starts agree");
    let bins = [(0, 30), (30, 100), (100, 300), (300, 1000), (1000, 3000), (3000, usize::MAX)];
    for (lo, hi) in bins {
        let mut v: Vec<&(usize, f64, GpResult)> = single.iter().filter(|r| r.0 >= lo && r.0 < hi).collect();
        if v.is_empty() {
            continue;
        }
        v.sort_by(|a, b| a.1.total_cmp(&b.1));
        let agree = v.iter().filter(|r| r.2.n_starts_agree == r.2.n_starts).count();
        let label = if hi == usize::MAX { format!("{lo}+") } else { format!("{lo}–{hi}") };
        println!(
            "{:>14}  {:>8}  {:>12.2}  {:>12.2}  {:>9.1}%",
            label,
            v.len(),
            1e3 * v[v.len() / 2].1,
            1e3 * v.last().unwrap().1,
            100.0 * agree as f64 / v.len() as f64
        );
    }

    let mut ls: Vec<f64> = single.iter().map(|r| r.2.hyper.length_scale).collect();
    ls.sort_by(f64::total_cmp);
    let q = |f: f64| ls[((ls.len() - 1) as f64 * f) as usize];
    println!("\nFitted length scale (days): p10 {:.2}  median {:.2}  p90 {:.2}", q(0.1), q(0.5), q(0.9));
}
