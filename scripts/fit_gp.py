#!/usr/bin/env python3
"""Fit a multi-band GP to photometry CSVs, extract light-curve features, plot.

Writes <output-dir>/features.csv (one row per source and band: peak time/mag,
t_rise, t_fade, FWHM, rise/fade rate in mag/day, each with a 1σ error),
<output-dir>/thermal.csv (one row per source: blackbody log T at peak and
cooling rate d log10 T/dt in dex/day over 30 d after peak, with errors), and a
plot per source with the peak (star) and FWHM (bar at half maximum) marked.

Usage:
    python scripts/fit_gp.py path/to/source.csv [more.csv ...] [--flux]
    python scripts/fit_gp.py path/to/photometry_dir/ --max-files 20

Plots are written to <repo>/gp_results/ by default. Requires the extension:
    maturin develop --release --features python
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import rust_gp

REPO = Path(__file__).resolve().parent.parent
BAND_COLORS = {"u": "#7b3fa0", "g": "#2a9d4b", "r": "#d62728", "i": "#8c564b", "z": "#555555", "y": "#e3a21a"}
ZP = 23.9


def flux_to_mag(f):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(f > 0, ZP - 2.5 * np.log10(f), np.nan)


FEATURES = ["peak_time", "peak_mag", "t_rise", "t_fade", "fwhm", "rise_rate", "fade_rate"]
HALF_MAX_MAG = 2.5 * math.log10(2)


def plot_fit(res, title, out_path, flux_space):
    det = [o for o in res["obs"] if not o.get("upper_limit")]
    lim = [o for o in res["obs"] if o.get("upper_limit")]
    t_det = np.array([o["time"] for o in det])
    t_ref = t_det.min()
    span = max(t_det.max() - t_ref, 1.0)
    # Frame on the detections, with room for nearby non-detections.
    pad = max(0.1 * span, 10.0)
    lo, hi = t_ref - pad, t_det.max() + pad
    ell = res["params"]["length_scale"]
    n_grid = int(min(20000, max(1000, 10 * (hi - lo) / ell)))
    t_grid = np.linspace(lo, hi, n_grid)
    tg = t_grid - t_ref
    feats = res.get("features", {})

    fig, ax = plt.subplots(figsize=(10, 5))
    for band in sorted({o["band"] for o in det}, key="ugrizy".index):
        color = BAND_COLORS.get(band, "k")
        sel = [o for o in det if o["band"] == band]
        t = np.array([o["time"] for o in sel]) - t_ref
        f = np.array([o["flux"] for o in sel])
        fe = np.array([o["flux_err"] for o in sel])
        # Non-detections: 5σ limit = 5 × the stored 1σ flux error.
        lsel = [o for o in lim if o["band"] == band and lo <= o["time"] <= hi]
        tl = np.array([o["time"] for o in lsel]) - t_ref
        fl = 5 * np.array([o["flux_err"] for o in lsel])
        mean, std = (np.array(a) for a in rust_gp.predict(res, t_grid.tolist(), band))
        if flux_space:
            ax.errorbar(t, f, fe, fmt="o", ms=3, color=color, alpha=0.6, label=f"{band} ({len(sel)})")
            ax.plot(tl, fl, "v", ms=5, color=color, alpha=0.5)
            ax.plot(tg, mean, color=color, lw=1.2)
            ax.fill_between(tg, mean - std, mean + std, color=color, alpha=0.2, lw=0)
        else:
            ax.errorbar(t, flux_to_mag(f), 2.5 / math.log(10) * fe / f, fmt="o", ms=3, color=color, alpha=0.6,
                        label=f"{band} ({len(sel)})")
            ax.plot(tl, flux_to_mag(fl), "v", ms=5, color=color, alpha=0.5)
            ax.plot(tg, flux_to_mag(mean), color=color, lw=1.2)
            ax.fill_between(tg, flux_to_mag(mean + std), flux_to_mag(np.maximum(mean - std, 1e-30)),
                            color=color, alpha=0.2, lw=0)
        annotate_features(ax, feats.get(band, {}), t_ref, color, flux_space)
    if not flux_space:
        # Where mean − σ ≤ 0 the faint edge of the band is at infinite
        # magnitude; frame the plot on the data instead (inverted axis).
        m_obs = flux_to_mag(np.array([o["flux"] for o in det]))
        ax.set_ylim(np.nanmax(m_obs) + 1.0, np.nanmin(m_obs) - 1.0)
    ax.set_xlim(lo - t_ref, hi - t_ref)
    p = res["params"]
    ax.set_title(
        f"{title}   ℓ={p['length_scale']:.2f} d, ℓ_λ={p['wavelength_scale']:.0f} Å, "
        f"{res['n_obs']} det + {res['n_limits']} limits, starts agree {res['n_starts_agree']}/{res['n_starts']}\n"
        + "   ".join(feature_summary(b, f) for b, f in feats.items() if f["peak_mag"] is not None),
        fontsize=8,
    )
    ax.set_xlabel(f"days since first detection ({t_ref:.3f})")
    ax.set_ylabel("flux (µJy)" if flux_space else "AB mag")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def annotate_features(ax, f, t_ref, color, flux_space):
    """Star at the peak; bar across the FWHM at half maximum."""
    if f.get("peak_mag") is None:
        return
    tp = f["peak_time"] - t_ref
    y_peak = f["peak_flux"] if flux_space else f["peak_mag"]
    y_half = 0.5 * f["peak_flux"] if flux_space else f["peak_mag"] + HALF_MAX_MAG
    ax.plot(tp, y_peak, "*", ms=12, color=color, mec="k", mew=0.6, zorder=5)
    x0 = tp - f["t_rise"] if f["t_rise"] is not None else tp
    x1 = tp + f["t_fade"] if f["t_fade"] is not None else tp
    if x1 > x0:
        ax.plot([x0, x1], [y_half, y_half], color=color, lw=2.2, zorder=4)
        for x, ok in ((x0, f["t_rise"] is not None), (x1, f["t_fade"] is not None)):
            if ok:
                ax.plot(x, y_half, "|", ms=12, mew=2.2, color=color, zorder=4)


def fmt(v, e, spec):
    if v is None:
        return "—"
    return f"{v:{spec}}±{e:{spec}}" if e is not None else f"{v:{spec}}"


def feature_summary(band, f):
    return (f"{band}: peak {fmt(f['peak_mag'], f['peak_mag_err'], '.2f')}, "
            f"rise {fmt(f['rise_rate'], f['rise_rate_err'], '.3f')}, fade {fmt(f['fade_rate'], f['fade_rate_err'], '.3f')} mag/d, "
            f"FWHM {fmt(f['fwhm'], f['fwhm_err'], '.1f')} d")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="CSV files and/or directories of CSVs")
    ap.add_argument("--output-dir", default=REPO / "gp_results", type=Path)
    ap.add_argument("--flux", action="store_true", help="plot in flux instead of magnitudes")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--n-starts", type=int, default=3)
    ap.add_argument("--no-limits", action="store_true", help="ignore non-detections (diffmaglim)")
    ap.add_argument("--no-plots", action="store_true", help="only write the features table")
    args = ap.parse_args()

    paths = []
    for inp in map(Path, args.inputs):
        paths += sorted(inp.glob("*.csv")) if inp.is_dir() else [inp]
    paths = [str(p) for p in paths[: args.max_files]]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = rust_gp.fit_many(paths, n_starts=args.n_starts, upper_limits=not args.no_limits)
    table = args.output_dir / "features.csv"
    with open(table, "w", newline="") as fh:
        cols = ["source", "band", "n_points", "peak_bracketed"] + [c for k in FEATURES for c in (k, f"{k}_err")]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for path, res in zip(paths, results):
            name = Path(path).stem
            if "error" in res:
                print(f"{name}: {res['error']}")
                continue
            for band, f in res["features"].items():
                w.writerow({"source": name, "band": band, **{c: f.get(c) for c in cols[2:]}})
            print(name + ":  " + "   ".join(feature_summary(b, f) for b, f in res["features"].items()
                                            if f["peak_mag"] is not None))
            if not args.no_plots:
                plot_fit(res, name, args.output_dir / f"{name}_gp.png", args.flux)
    with open(args.output_dir / "thermal.csv", "w", newline="") as fh:
        cols = ["source", "bands", "ref_band", "n_epochs", "log_temp_peak", "log_temp_peak_err",
                "cooling_rate", "cooling_rate_err", "log_temp_latest", "log_temp_latest_err"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for path, res in zip(paths, results):
            if "error" in res:
                continue
            th = res["thermal"]
            w.writerow({"source": Path(path).stem, "bands": "".join(th["bands"]), "n_epochs": len(th["times"]),
                        **{c: th[c] for c in cols[2:] if c in th}})
    print(f"features -> {table}\nthermal  -> {args.output_dir / 'thermal.csv'}")


if __name__ == "__main__":
    main()
