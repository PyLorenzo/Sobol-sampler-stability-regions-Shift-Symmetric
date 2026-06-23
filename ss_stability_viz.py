"""
Author: Lorenzo Baldazzi
Affiliation: University of Rome Tor Vergata
Date: 2026-06-04

Visualisation of the Shift Symmetric stability map produced by ss_stability_map.py.

Generates two figures:

(A) Corner-style SCATTER plot: each off-diagonal panel shows the labelled
    points projected on a (param_i, param_j) pair, coloured by stability.

(B) Corner-style MARGINAL-FRACTION heatmap: each off-diagonal panel shows the
    local stable fraction in 2-D bins, computed by averaging the binary
    stability label over the other parameters. This is the "probability
    of stability" map.

Usage
-----
    python ss_stability_viz.py ss_stability_map.pkl
    python ss_stability_viz.py ss_stability_map.pkl --bins 30 --outdir ./figs
"""

from __future__ import annotations

import argparse
import os
import pickle
from itertools import combinations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

LATEX = {
    "Shift_Symmetric_alphaB0": r"$\alpha_{\mathrm{B},0}$",
    "Shift_Symmetric_m":       r"$m$",
}


def load(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ----------------------------------------------------------------------
# (A) Scatter corner plot
# ----------------------------------------------------------------------

def scatter_corner(data: dict, outpath: str,
                   max_points: int = 20000,
                   point_size: float = 3):
    """
    Lower-triangular corner of scatter plots. Stable points in green, unstable
    in red. If there are more than max_points, a uniform random subsample is
    drawn so the figure stays readable.
    """
    names  = data["param_names"]
    ranges = data["param_ranges"]
    pts    = data["points"]
    stab   = data["stable"].astype(bool)
    d      = len(names)

    # Subsample if needed
    n = len(pts)
    if n > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, max_points, replace=False)
        pts, stab = pts[idx], stab[idx]

    fig, axes = plt.subplots(d, d, figsize=(2.6 * d, 2.6 * d),
                             squeeze=False)

    for i in range(d):
        for j in range(d):
            ax = axes[i, j]

            if j > i:
                ax.axis("off")
                continue

            if i == j:
                # 1-D marginal: histogram of all points (=prior) vs stable
                lo, hi = ranges[names[i]]
                bins = np.linspace(lo, hi, 40)
                ax.hist(pts[:, i], bins=bins, color="0.7",
                        label="all", alpha=0.7)
                ax.hist(pts[stab, i], bins=bins, color="C2",
                        label="stable", alpha=0.8)
                ax.set_yticks([])
                ax.set_xlim(lo, hi)
            else:
                # 2-D scatter
                ax.scatter(pts[~stab, j], pts[~stab, i],
                           s=point_size, c="#d62728", alpha=0.55,
                           edgecolors="none", rasterized=True,
                           label="unstable")
                ax.scatter(pts[ stab, j], pts[ stab, i],
                           s=point_size, c="#2ca02c", alpha=0.55,
                           edgecolors="none", rasterized=True,
                           label="stable")
                ax.set_xlim(ranges[names[j]])
                ax.set_ylim(ranges[names[i]])

            # axis labels only on the outside
            if i == d - 1:
                ax.set_xlabel(LATEX.get(names[j], names[j]))
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(LATEX.get(names[i], names[i]))
            else:
                ax.set_yticklabels([])

    fig.tight_layout(rect=[0, 0, 0.92, 1.0])
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='0.7', markersize=8,
               alpha=0.7, label='all'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='C2', markersize=8,
               alpha=0.8, label='stable'),
        Line2D([0], [0], marker='o', color='#d62728', markerfacecolor='#d62728',
               linestyle='None', markersize=6, alpha=0.75, label='unstable'),
    ]
    fig.legend(handles=legend_handles,
               loc='upper right',
               bbox_to_anchor=(0.98, 0.98),
               borderaxespad=0.1,
               fontsize=8)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  wrote {outpath}")


# ----------------------------------------------------------------------
# (B) Marginalised stability-fraction heatmap
# ----------------------------------------------------------------------

def marginal_fraction_corner(data: dict, outpath: str,
                             bins: int = 25,
                             min_count: int = 3):
    """
    Each off-diagonal panel shows P(stable | param_i, param_j) estimated by
    binning the points in 2-D and computing

           p_hat(i,j)  =  sum stable label in bin / sum points in bin

    Bins with fewer than `min_count` points are masked (white) to avoid
    over-confidence in undersampled regions.

    Diagonals: 1-D version (P(stable | param_i)) plus a histogram of the
    sample density.
    """
    names  = data["param_names"]
    ranges = data["param_ranges"]
    pts    = data["points"]
    stab   = data["stable"].astype(float)        # 0/1, makes mean = fraction
    d      = len(names)

    fig, axes = plt.subplots(d, d, figsize=(2.7 * d, 2.7 * d), squeeze=False)
    cmap = plt.get_cmap("RdYlGn")
    cmap.set_bad("lightgrey")

    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue

            xi_lo, xi_hi = ranges[names[i]]

            if i == j:
                # 1-D marginal: P(stable | x_i)
                xbins = np.linspace(xi_lo, xi_hi, bins + 1)
                count_total,  _ = np.histogram(pts[:, i], bins=xbins)
                count_stable, _ = np.histogram(pts[:, i], bins=xbins,
                                               weights=stab)
                with np.errstate(invalid="ignore", divide="ignore"):
                    frac = np.where(count_total >= min_count,
                                    count_stable / count_total,
                                    np.nan)
                centers = 0.5 * (xbins[1:] + xbins[:-1])
                ax.bar(centers, frac, width=xbins[1] - xbins[0],
                       color=cmap(np.nan_to_num(frac, nan=0.5)),
                       edgecolor="none")
                ax.set_ylim(-0.02, 1.02)
                ax.set_xlim(xi_lo, xi_hi)
                ax.set_yticks([0, 0.5, 1.0])
                if i == 0:
                    ax.set_ylabel("P(stable)", fontsize=8)
            else:
                # 2-D heatmap: P(stable | x_j, x_i)
                xj_lo, xj_hi = ranges[names[j]]
                xbins = np.linspace(xj_lo, xj_hi, bins + 1)
                ybins = np.linspace(xi_lo, xi_hi, bins + 1)

                count_total,  _, _ = np.histogram2d(
                    pts[:, j], pts[:, i], bins=[xbins, ybins])
                count_stable, _, _ = np.histogram2d(
                    pts[:, j], pts[:, i], bins=[xbins, ybins], weights=stab)

                with np.errstate(invalid="ignore", divide="ignore"):
                    frac = np.where(count_total >= min_count,
                                    count_stable / count_total,
                                    np.nan)
                frac = np.ma.masked_invalid(frac)

                X, Y = np.meshgrid(xbins, ybins)
                im = ax.pcolormesh(X, Y, frac.T,
                                   cmap=cmap,
                                   vmin=0.0, vmax=1.0,
                                   shading="auto")
                ax.set_xlim(xj_lo, xj_hi)
                ax.set_ylim(xi_lo, xi_hi)

            if i == d - 1:
                ax.set_xlabel(LATEX.get(names[j], names[j]))
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(LATEX.get(names[i], names[i]))
            else:
                if j != 0:
                    ax.set_yticklabels([])

    # One shared colorbar on the right
    fig.subplots_adjust(right=0.90)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.018, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("P(stable)  — marginalised over the other parameters")

    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pickle_path",
                    help="Pickle written by ss_stability_map.py")
    ap.add_argument("--bins", type=int, default=25,
                    help="Bins per axis for the marginal heatmap "
                         "(default: %(default)s)")
    ap.add_argument("--min-count", type=int, default=3,
                    help="Bins with fewer than this many points are masked "
                         "(default: %(default)s)")
    ap.add_argument("--max-scatter", type=int, default=20000,
                    help="Subsample for the scatter corner (default: %(default)s)")
    ap.add_argument("--outdir", default=".",
                    help="Output directory for figures (default: %(default)s)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    data = load(args.pickle_path)

    print(f"Loaded {len(data['points'])} points "
          f"(stable fraction = {data['stable'].mean():.3f})")

    scatter_path = os.path.join(args.outdir, "ss_stability_scatter.png")
    heatmap_path = os.path.join(args.outdir, "ss_stability_heatmap.png")

    scatter_corner(data, scatter_path,
                   max_points=args.max_scatter)
    marginal_fraction_corner(data, heatmap_path,
                             bins=args.bins,
                             min_count=args.min_count)


if __name__ == "__main__":
    main()
