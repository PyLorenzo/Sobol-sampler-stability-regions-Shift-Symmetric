# Sobol sampler for the stability regions of the Shift-Symmetric model

A two-stage toolkit that maps and visualises the **viable (stable) region** of the
Shift-Symmetric modified-gravity model as implemented in
[EFTCAMB](https://eftcamb.org/), in the two-dimensional parameter plane

$\theta = (\alpha_{B,0},m)$

The package contains two scripts:

| Script | Role |
| --- | --- |
| `ss_stability_map.py` | Samples the parameter box, asks EFTCAMB whether each point is stable, and writes a labelled point cloud to a pickle file. |
| `ss_stability_viz.py` | Reads that pickle and produces two corner-style figures: a raw scatter map and a marginalised stability-probability heatmap. |

---

## 1. Physical background

The Shift-Symmetric model parametrises the **braiding** function of the
Horndeski/α-basis as

$$
\alpha_B(a) = \alpha_{B,0} \left(\frac{H_0}{H(a)}\right)^{4/m}
$$

with two free constants, $\alpha_{B,0}$ and $m$ (the parametrisation proposed by
Traykova et al. 2021 for shift-symmetric scalar-tensor theories on a CPL/ $w_0 w_a CDM$
background). The remaining α-function $\alpha_K$ is set by a fiducial $\alpha_{K,0}$, and
the background equation of state is fixed through $w_0$, $w_a$.

For each choice of $(\alpha_{B,0}, m)$, EFTCAMB evolves the linear perturbations and
checks, at every time step and every scale up to $k_{\mathrm{max}}$, a set of **viability
conditions**:

- **no-ghost**: the kinetic term of the extra scalar degree of freedom is positive
  ($Q_s > 0$);
- **no-gradient instability**: the scalar sound speed squared is positive
  ($c_s^2 > 0$);
- **mathematical (classical) stability** of the $\pi$-field equation (well-posed
  leading coefficient, no fast exponential growth), plus a well-defined tensor
  equation.

If any condition fails at any redshift or scale, EFTCAMB flags the point as
**unstable**. The set of points that pass *all* active conditions forms the stability
region. Its boundary in $(\alpha_{B,0}, m)$ is **not known analytically**, which is why
it is mapped numerically here.

> The stability flags that are enforced are exactly those configured in the user's
> EFTCAMB / Cobaya setup (`EFT_mathematical_stability`, `EFT_physical_stability`,
> `EFT_additional_priors`, …). This code does not impose its own conditions — it
> simply records EFTCAMB's verdict.

---

## 2. How `ss_stability_map.py` works

The script never runs a full MCMC and never evaluates a single CMB power spectrum.
It reduces the cosmology code to the **cheapest object that still triggers the
stability module**, then samples that object efficiently. The pipeline has five
conceptual stages.

### 2.1 Build a minimal Cobaya model (`build_cobaya_info`, `make_model`)

The user's Cobaya YAML is read and stripped down:

- the `theory` block (CAMB + EFTCAMB + all `extra_args`) is copied verbatim, so the
  Shift-Symmetric parametrisation and all stability flags are inherited unchanged;
- every likelihood is replaced by the no-op `one` likelihood. With `one`,
  `loglike == 0` **if and only if** all theory components initialise successfully, and
  `-inf` otherwise — so the *only* thing that can send the log-likelihood to `-inf` is
  EFTCAMB's internal stability check;
- the six $\Lambda CDM$ parameters ($\Omega_b^2$,$\Omega_c^2$, $tau$, $logA$, $n_s$, $H_0$) are pinned at the
  centres of their reference distributions. They are declared as "sampled" with a
  trivial uniform prior solely so that Cobaya accepts them as inputs — the prior is
  never consulted;
- `alphaK0`, `EFTw0`, `EFTwa` are declared with `value:` in the YAML and handled as
  fixed constants by Cobaya;
- the YAML's derived-parameter definitions (e.g. $A_s$ <- $logA$) are carried over so
  CAMB still receives the inputs it expects.

**Why:** a full power-spectrum evaluation costs of order seconds per point; the bare
stability check costs tens of milliseconds. Over thousands of points this is a one-
to-two order of magnitude speed-up.

### 2.2 The stability oracle (`is_stable`)

For a parameter point $\theta$,

```python
ll = model.loglike(theta, make_finite=False, cached=False, return_derived=False)
return np.isfinite(ll)
```

`make_finite=False` lets the `-inf` produced by an unstable model propagate to Python
instead of being clamped, so `np.isfinite` cleanly separates stable (`True`) from
unstable (`False`). Any unexpected exception (NaN propagation, a CAMB/Fortran error)
is caught and treated as **unstable** — the code errs on the safe side.

### 2.3 Sobol quasi-Monte-Carlo sampling (`sobol_points`)

The base sample is drawn from a **scrambled Sobol low-discrepancy sequence** rather
than pseudo-random Monte Carlo.

- A Sobol sequence in base 2 is a $(0, m, 2)$-net at $N = 2^m$: every dyadic sub-box
  of the unit square with area $2^{-m}$ contains *exactly one* point. The unit square
  is therefore covered with no gaps and no clumps.
- The star discrepancy of such a net scales as $O(\log N / N)$, versus $O(N^{-1/2})$
  for pseudo-random points. By the Koksma–Hlawka inequality the integration / coverage
  error is bounded by $V_{\mathrm{HK}}(f) \cdot D^*_N$, so a smaller discrepancy directly means a more
  accurate estimate of the stable volume fraction at fixed $N$.
- The net property is exact only when $N$ is a power of two, so the requested sample
  size $n$ is rounded **up** to $2^{\lceil\log_2 n\rceil}$, generated, and then truncated to $n$.
  This keeps the strongest available balance within the caller's budget.
- `scramble=True` (Owen scrambling) makes the estimator unbiased and reproducible from
  a fixed seed, without degrading the net property.

The unit-cube points are mapped linearly onto the physical prior box,
$$
\theta_j = \mathrm{lo}_j + (\mathrm{hi}_j - \mathrm{lo}_j) \cdot u_j
$$

### 2.4 Parallel evaluation (`evaluate_points`, `_worker_init`, `_worker_eval`)

EFTCAMB wraps compiled Fortran that keeps internal state, which dictates the
parallelisation strategy:

- **`spawn` start method** (set in `__main__`): each worker process starts fresh, so
  no Fortran heap is shared between processes. `fork` would copy and then corrupt that
  state.
- **one model per worker, built once**: `get_model()` is expensive, so each worker
  builds it a single time in `_worker_init` and stores it in a module global; every
  subsequent point reuses it.
- **`maxtasksperchild=200`**: workers are recycled periodically to cap the slow memory
  growth of CAMB's Fortran heap.
- a serial fallback (`--serial`, or `--workers 1`) runs everything in-process, which is
  the right mode for debugging tracebacks.

### 2.5 Boundary-driven adaptive refinement (`boundary_uncertainty`, `refine_box`)

After the global sample is labelled, the script concentrates additional points near
the stability boundary, where the map is most informative.

**Uncertainty score.** Coordinates are normalised to $[0,1]^2$ so both parameters
weigh equally in the distance metric. For each point the fraction $f_i$ of stable
points among its $k$ nearest neighbours (k-NN, default $k = 8$) is computed, and

$$
u_i = 1 - \left\lvert 2 f_i - 1 \right\rvert
$$

This is a tent function of $f_i$: $\nu_i = 0$ in a homogeneous neighbourhood (all stable
or all unstable) and $\nu_i = 1$ when exactly half the neighbours are stable — the
signature of a point sitting on the boundary. The score is fully non-parametric; no
shape is assumed for the boundary.

**Refinement box.** The points in the top decile of uncertainty ($\text{top\_quantile} = 0.9$)
define an axis-aligned bounding box, which is padded by $5\%$ on each side, clipped back
to the physical prior box, and then re-sampled with a fresh Sobol batch. If fewer than
four boundary points exist (too early to localise a boundary), the code falls back to a
global Sobol draw. This repeats `--refine-iters` times.

### 2.6 Output

A single pickle is written with the following schema:

```python
{
  "param_names":  ["Shift_Symmetric_alphaB0", "Shift_Symmetric_m"],
  "param_ranges": {name: (lo, hi), ...},
  "ss_fixed":     {"$\Lambda$CDM name": value, ...},
  "points":       ndarray shape (N, 2),   # all evaluated points
  "stable":       ndarray shape (N,),     # bool labels
  "iter_info":    [ {"phase", "n", "stable_frac"}, ... ],
  "yaml":         absolute path to the source YAML,
}
```

---

## 3. How `ss_stability_viz.py` works

Reads the pickle and produces two corner-style PNGs. With a single 2-D parameter pair
the "corner" is a $2\times 2$ grid: two 1-D marginals on the diagonal and one off-diagonal
panel for the $(\alpha_{B,0}, m)$ plane.

### 3.1 Scatter corner (`scatter_corner` → `ss_stability_scatter.png`)

- **Off-diagonal:** every labelled point plotted in the parameter plane, stable in
  green and unstable in red. If the cloud exceeds `--max-scatter` (default 20 000), a
  uniform random subsample is drawn so the figure stays legible — note this is purely
  cosmetic and does not affect any computed quantity.
- **Diagonal:** a 1-D histogram of *all* points (grey, ≈ the sampling/prior density)
  overlaid with the histogram of the *stable* points (green).

This is the most direct view: it shows the actual sampling density, makes the
refinement passes visible as a denser band of points hugging the boundary, and reveals
the raw shape of the stable set.

### 3.2 Marginalised stability-probability heatmap (`marginal_fraction_corner` → `ss_stability_heatmap.png`)

Each off-diagonal panel estimates the local stable fraction by binning the points in
2-D and averaging the binary label:

$$
\hat{p}(\mathrm{bin}) = \frac{\text{number of stable points in bin}}{\text{total points in bin}}
$$

Because `stable` is cast to `float` (0/1), the bin mean *is* the empirical
$P(\text{stable} \mid \alpha_{B,0}, m)$. The diagonal shows the 1-D version,
$P(\text{stable} \mid x_i)$. Two safeguards keep the estimate honest:

- bins with fewer than `--min-count` points (default 3) are **masked** (shown grey)
  rather than reported, to avoid over-confident fractions from undersampled cells;
- the colour scale is fixed to $[0, 1]$ with a single shared colour bar, so panels are
  directly comparable.

The colour map is `RdYlGn` (red = unstable, green = stable), with masked/undersampled
cells in light grey.

> **Interpretation note.** With adaptive refinement the sampling density is *not*
> uniform — points are deliberately concentrated near the boundary. The heatmap fraction
> is still a valid *local* estimate of $P(\text{stable})$ within each bin, but the diagonal
> 1-D "$P(\text{stable})$" curves are conditioned on the (non-uniform) sampling and should not be
> read as a marginalisation over a flat prior. For an unbiased marginal, regenerate the
> map with `--refine-iters 0` (pure global Sobol).

---

## 4. Installation

The scripts require a working **EFTCAMB + CAMB + Cobaya** installation, plus a small
set of scientific-Python packages.

```bash
# (1) EFTCAMB / CAMB / Cobaya — follow the upstream EFTCAMB install guide:
#     https://eftcamb.org/  and  https://cobaya.readthedocs.io/

# (2) Python dependencies for this toolkit:
pip install -r requirements.txt
```

`requirements.txt` covers `numpy`, `scipy`, `scikit-learn`, `matplotlib`, and
`pyyaml`. Cobaya and CAMB/EFTCAMB are intentionally *not* listed there, since they have
their own (compiler-dependent) installation procedure.

---

## 5. Usage

### Map the stability region

```bash
# quick smoke test (~1 minute on 4 cores)
python ss_stability_map.py --yaml /path/to/Shift_Symmetric.yaml \
                           --base 64 --refine-iters 0 \
                           --output ss_stability_map.pkl

# full run (tens of minutes to hours, depending on cores)
python ss_stability_map.py --yaml /path/to/Shift_Symmetric.yaml \
                           --base 4096 --refine-iters 3 --refine-batch 1024 \
                           --output ss_stability_map.pkl
```

Key options:

| Flag | Meaning | Default |
| --- | --- | --- |
| `--yaml` | Cobaya YAML used as the source of the theory block | (edit in script) |
| `--base` | base Sobol sample size (rounded up to a power of 2) | 4096 |
| `--refine-iters` | number of boundary-refinement passes (0 disables) | 3 |
| `--refine-batch` | points added per refinement pass | 1024 |
| `--workers` | parallel worker processes | `ncpu - 1` |
| `--seed` | Sobol seed (reproducibility) | 42 |
| `--serial` | run in-process (debugging) | off |
| `--output` | output pickle path | (edit in script) |

### Visualise

```bash
python ss_stability_viz.py ss_stability_map.pkl --bins 30 --outdir ./figs
```

| Flag | Meaning | Default |
| --- | --- | --- |
| `--bins` | bins per axis for the heatmap | 25 |
| `--min-count` | mask bins with fewer than this many points | 3 |
| `--max-scatter` | subsample size for the scatter figure | 20000 |
| `--outdir` | output directory for the PNGs | `.` |

Outputs: `ss_stability_scatter.png` and `ss_stability_heatmap.png`.

---

## 6. Repository layout

```
Sobol-sampler-stability-regions-Shift-Symmetric/
├── README.md
├── requirements.txt
├── .gitignore
├── ss_stability_map.py     # sampling + stability evaluation -> pickle
└── ss_stability_viz.py     # pickle -> scatter + probability figures
```

---

## 7. References

- D. Traykova et al., *Theoretical priors in scalar-tensor cosmologies:
  Shift-symmetric Horndeski models*, Phys. Rev. D **104**, 083502 (2021),
  arXiv:2103.11195 — the `alpha_B(a)` parametrisation used here.
- B. Hu, M. Raveri, N. Frusciante, A. Silvestri, *EFTCAMB/EFTCosmoMC: Numerical Notes*,
  arXiv:1405.3590 — EFTCAMB stability module and flags.
- N. Frusciante, L. Perenon, *Effective field theory of dark energy: A review*,
  Phys. Rep. **857**, 1 (2020), arXiv:1907.03150.
- H. Niederreiter, *Random Number Generation and Quasi-Monte Carlo Methods*, SIAM
  (1992) — `(t,m,d)`-nets, discrepancy, the Koksma–Hlawka inequality.
- S. Joe, F. Y. Kuo, *Constructing Sobol sequences with better two-dimensional
  projections*, SIAM J. Sci. Comput. **30**, 2635 (2008).
- A. B. Owen, *Randomly permuted (t,m,s)-nets and (t,s)-sequences*, in *Monte Carlo
  and Quasi-Monte Carlo Methods in Scientific Computing* (1995) — scrambling.

---

## 8. Author

Lorenzo Baldazzi — University of Rome Tor Vergata.
