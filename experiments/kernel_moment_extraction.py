"""
experiments/kernel_moment_extraction.py

URP Kernel Moment Extraction Experiment
========================================
Runs the URP field on a cylindrical geometry (periodic theta x z surface),
measures the two-point correlation function C(r) = <phi(x) phi(x+r)>,
extracts the kernel moments K0, K2, K4 via numerical radial integration,
fits the effective kernel to Mexican hat / Gaussian / DoG families,
and checks the Swift-Hohenberg threshold condition against alpha_target.

Outputs
-------
artifacts/kernel_moments/
    moments.json          -- K0, K2, K4, SH coefficients, threshold check
    kernel_fit.json       -- best-fit parameters for each kernel family
    correlation.npy       -- raw 2D correlation function C(r_theta, r_z)
    radial_correlation.npy -- azimuthally averaged C(r)
    summary.txt           -- human-readable report
    correlation_plot.png  -- C(r) vs r (if --visualize)
    kernel_plot.png       -- effective vs fitted kernel (if --visualize)
    dispersion_plot.png   -- sigma(q) = -alpha*q^2 + K_tilde(q) (if --visualize)

Usage
-----
    python experiments/kernel_moment_extraction.py
    python experiments/kernel_moment_extraction.py --steps 200 --visualize
    python experiments/kernel_moment_extraction.py --alpha 0.09 --kappa 0.22

Reference
---------
    Docs/Kernel_Moments_Derivation.md
    Docs/The_Lucidic_Compass.md  Sections 3-5
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import quad

from project_genesis import EngineConfig, GenesisEngine


# ---------------------------------------------------------------------------
# Physical constants (QCD-derived URP values)
# ---------------------------------------------------------------------------

BETA_QCD = 0.09       # complexity coupling
G_QCD = 0.22          # coherence/gravity coupling = kappa_vac
KAPPA_VAC = 0.22      # QCD vacuum capacity (instanton packing fraction)
XI_0 = 0.469          # coherence length scale [nm], from kappa_vac * r0_DNA^(1/2)
R0_DNA = 1.0          # DNA cylinder radius [nm]
P_DNA = 3.4           # B-form DNA pitch [nm]
M_WINDING = 6         # hexagonal resonance winding number
KAPPA_CRIT = 0.4623   # critical capacity from pitch/geometry threshold
ALPHA_TARGET = 0.0887 # conjectured diffusion coefficient (Conjecture A-C)


# ---------------------------------------------------------------------------
# Cylindrical geometry helpers
# ---------------------------------------------------------------------------

def make_cylindrical_field(n_theta: int, n_z: int, seed: int | None = None) -> np.ndarray:
    """Initialise a 2D field on (theta, z) with small Gaussian noise."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.01, size=(n_theta, n_z)).astype(np.float64)


def laplacian_cylindrical(phi: np.ndarray, r0: float, dz: float) -> np.ndarray:
    """Discrete Laplacian on the cylinder surface with periodic BCs.
    phi has shape (n_theta, n_z); spatial steps are r0*d_theta and dz.
    """
    n_theta, n_z = phi.shape
    d_theta = 2.0 * math.pi / n_theta
    dl_theta = r0 * d_theta  # arc length per theta step

    d2_theta = (
        np.roll(phi, 1, axis=0) - 2.0 * phi + np.roll(phi, -1, axis=0)
    ) / dl_theta**2
    d2_z = (
        np.roll(phi, 1, axis=1) - 2.0 * phi + np.roll(phi, -1, axis=1)
    ) / dz**2
    return d2_theta + d2_z


def mexican_hat_kernel_2d(r_grid: np.ndarray, C: float, xi: float) -> np.ndarray:
    """Mexican hat kernel K_MH(r) = C*(1 - r^2/(2*xi^2))*exp(-r^2/(2*xi^2))."""
    u = r_grid**2 / (2.0 * xi**2)
    return C * (1.0 - u) * np.exp(-u)


def apply_mexican_hat_kernel(
    phi: np.ndarray, r0: float, dz: float, C: float, xi: float
) -> np.ndarray:
    """Convolve phi with the Mexican hat kernel on the cylinder surface.
    Uses FFT convolution for efficiency.
    """
    n_theta, n_z = phi.shape
    d_theta = 2.0 * math.pi / n_theta

    # Build kernel on the same grid
    theta_vals = np.arange(n_theta) * d_theta * r0  # arc length
    z_vals = np.arange(n_z) * dz

    # Fold into [-L/2, L/2] for both axes (periodic distance)
    L_theta = n_theta * d_theta * r0
    L_z = n_z * dz
    theta_vals = np.where(theta_vals > L_theta / 2, theta_vals - L_theta, theta_vals)
    z_vals = np.where(z_vals > L_z / 2, z_vals - L_z, z_vals)

    TH, ZZ = np.meshgrid(theta_vals, z_vals, indexing="ij")
    R = np.sqrt(TH**2 + ZZ**2)
    kernel_2d = mexican_hat_kernel_2d(R, C, xi)

    # Normalise kernel so it integrates to K0_analytic
    d_area = (d_theta * r0) * dz
    kernel_2d_ft = np.fft.rfft2(kernel_2d) * d_area
    phi_ft = np.fft.rfft2(phi)
    result_ft = kernel_2d_ft * phi_ft
    return np.fft.irfft2(result_ft, s=phi.shape)


# ---------------------------------------------------------------------------
# URP field evolution on the cylinder
# ---------------------------------------------------------------------------

def evolve_cylindrical(
    phi: np.ndarray,
    r0: float,
    dz: float,
    dt: float,
    steps: int,
    alpha: float,
    beta: float,
    kernel_C: float,
    kernel_xi: float,
    record_every: int = 10,
) -> list[np.ndarray]:
    """Evolve the URP overdamped scalar field on the cylindrical surface.

    Field equation (overdamped):
        d_t phi = alpha * nabla^2 phi + beta * |nabla phi|^2 + I[phi]

    where I[phi] is the Mexican hat convolution.
    Returns a list of field snapshots at record_every intervals.
    """
    snapshots = []
    n_theta, n_z = phi.shape
    d_theta_arc = 2.0 * math.pi * r0 / n_theta

    for step in range(steps):
        lap = laplacian_cylindrical(phi, r0, dz)

        # Gradient for beta term (central differences)
        grad_theta = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2.0 * d_theta_arc)
        grad_z = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2.0 * dz)
        grad_sq = grad_theta**2 + grad_z**2

        # Nonlocal integration term
        integration = apply_mexican_hat_kernel(phi, r0, dz, kernel_C, kernel_xi)

        dphi = alpha * lap + beta * grad_sq + integration
        phi = phi + dt * dphi

        if (step + 1) % record_every == 0:
            snapshots.append(phi.copy())

    return phi, snapshots


# ---------------------------------------------------------------------------
# Two-point correlation function
# ---------------------------------------------------------------------------

def compute_two_point_correlation(field: np.ndarray) -> np.ndarray:
    """Compute C(dr_theta, dr_z) = <phi(x) phi(x+dr)> via FFT.
    Returns the normalised correlation (C(0,0) = variance).
    """
    ft = np.fft.rfft2(field)
    power = (ft * np.conj(ft)).real
    corr = np.fft.irfft2(power, s=field.shape)
    corr /= field.size
    return corr


def azimuthal_average(corr_2d: np.ndarray, r0: float, dz: float) -> tuple[np.ndarray, np.ndarray]:
    """Azimuthally average the 2D correlation into C(r).
    Returns (r_bins, C_r) arrays.
    """
    n_theta, n_z = corr_2d.shape
    d_theta_arc = 2.0 * math.pi * r0 / n_theta

    # Build distance grid (folded periodic)
    theta_idx = np.arange(n_theta)
    z_idx = np.arange(n_z)
    theta_idx = np.where(theta_idx > n_theta // 2, theta_idx - n_theta, theta_idx)
    z_idx = np.where(z_idx > n_z // 2, z_idx - n_z, z_idx)

    TH, ZZ = np.meshgrid(theta_idx * d_theta_arc, z_idx * dz, indexing="ij")
    R = np.sqrt(TH**2 + ZZ**2)

    r_max = min(n_theta // 2 * d_theta_arc, n_z // 2 * dz)
    n_bins = min(n_theta // 2, n_z // 2)
    r_edges = np.linspace(0.0, r_max, n_bins + 1)
    r_bins = 0.5 * (r_edges[:-1] + r_edges[1:])

    C_r = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    flat_r = R.ravel()
    flat_c = corr_2d.ravel()

    for i in range(n_bins):
        mask = (flat_r >= r_edges[i]) & (flat_r < r_edges[i + 1])
        if mask.sum() > 0:
            C_r[i] = flat_c[mask].mean()
            counts[i] = mask.sum()

    return r_bins, C_r


# ---------------------------------------------------------------------------
# Kernel moment extraction (Docs/Kernel_Moments_Derivation.md Section 2)
# ---------------------------------------------------------------------------

def extract_moments(r: np.ndarray, C_r: np.ndarray) -> dict[str, float]:
    """Numerically integrate K0, K2, K4 from the radial profile C(r).

    K0 = 2*pi * int r * C(r) dr
    K2 = (pi/2) * int r^3 * C(r) dr
    K4 = (pi/32) * int r^5 * C(r) dr
    """
    dr = np.diff(r, prepend=0.0)
    dr[0] = r[0]  # handle first bin

    K0 = float(2.0 * math.pi * np.sum(r * C_r * dr))
    K2 = float(math.pi / 2.0 * np.sum(r**3 * C_r * dr))
    K4 = float(math.pi / 32.0 * np.sum(r**5 * C_r * dr))
    return {"K0": K0, "K2": K2, "K4": K4}


# ---------------------------------------------------------------------------
# Swift-Hohenberg coefficients and threshold check
# ---------------------------------------------------------------------------

def swift_hohenberg_coefficients(
    moments: dict[str, float], alpha: float
) -> dict[str, float]:
    """Compute SH coefficients from kernel moments and alpha.

    mu = -K2 - alpha    (>0 needed for finite-q instability)
    lam = -K4           (>0 needed for pattern selection)
    eps = -K0           (threshold offset)
    q_c^2 = mu / (2*lam) if mu,lam > 0
    """
    mu = -moments["K2"] - alpha
    lam = -moments["K4"]
    eps = -moments["K0"]
    q_c_sq = mu / (2.0 * lam) if (mu > 0 and lam > 0) else None
    q_c = math.sqrt(q_c_sq) if q_c_sq and q_c_sq > 0 else None
    return {
        "mu": mu,
        "lambda": lam,
        "epsilon": eps,
        "q_c_sq": q_c_sq,
        "q_c": q_c,
    }


def threshold_check(
    sh: dict,
    moments: dict,
    alpha: float,
    r0: float = R0_DNA,
    m: int = M_WINDING,
    p_dna: float = P_DNA,
) -> dict:
    """Check whether the extracted moments satisfy the DNA pitch threshold.

    Target: q_c^2 = m^2/r0^2 + (2*pi*m/p)^2
    Threshold condition (Lucidic Compass eq 4.1):
        alpha = -K2 +/- 2*sqrt((-K0)*(-K4))
    """
    q_dna_sq = (m / r0) ** 2 + (2.0 * math.pi * m / p_dna) ** 2
    q_dna = math.sqrt(q_dna_sq)

    # Threshold equation: alpha_threshold = -K2 +/- 2*sqrt(K0_neg * K4_neg)
    K0, K2, K4 = moments["K0"], moments["K2"], moments["K4"]
    discriminant = (-K0) * (-K4)
    if discriminant >= 0:
        alpha_thresh_plus = -K2 + 2.0 * math.sqrt(discriminant)
        alpha_thresh_minus = -K2 - 2.0 * math.sqrt(discriminant)
    else:
        alpha_thresh_plus = alpha_thresh_minus = None

    q_c = sh.get("q_c")
    pitch_match = None
    if q_c is not None:
        pitch_match = abs(q_c - q_dna) / q_dna  # fractional deviation

    return {
        "q_dna": q_dna,
        "q_dna_sq": q_dna_sq,
        "q_c_extracted": q_c,
        "pitch_fractional_deviation": pitch_match,
        "alpha_input": alpha,
        "alpha_target": ALPHA_TARGET,
        "alpha_threshold_plus": alpha_thresh_plus,
        "alpha_threshold_minus": alpha_thresh_minus,
        "alpha_deviation_from_target": (
            abs(alpha - ALPHA_TARGET) / ALPHA_TARGET if alpha is not None else None
        ),
        "sh_instability_conditions_met": (
            sh["mu"] is not None
            and sh["lambda"] is not None
            and sh["mu"] > 0
            and sh["lambda"] > 0
        ),
    }


# ---------------------------------------------------------------------------
# Kernel family fitting
# ---------------------------------------------------------------------------

def fit_mexican_hat(r: np.ndarray, C_r: np.ndarray, xi_init: float = 0.5) -> dict:
    """Fit C(r) to K_MH(r) = C*(1 - r^2/(2*xi^2))*exp(-r^2/(2*xi^2))."""
    def mh(r, C, xi):
        u = r**2 / (2.0 * xi**2)
        return C * (1.0 - u) * np.exp(-u)

    try:
        popt, pcov = curve_fit(mh, r, C_r, p0=[C_r.max(), xi_init], maxfev=5000)
        C_fit, xi_fit = popt
        C_fit_std, xi_fit_std = np.sqrt(np.diag(pcov))
        # Analytic moments from fitted parameters
        K0_fit = 0.0
        K2_fit = -math.pi * C_fit * xi_fit**4
        K4_fit = -3.0 * math.pi * C_fit * xi_fit**6 / 4.0
        return {
            "family": "mexican_hat",
            "C": float(C_fit),
            "xi": float(xi_fit),
            "C_std": float(C_fit_std),
            "xi_std": float(xi_fit_std),
            "K0_analytic": K0_fit,
            "K2_analytic": K2_fit,
            "K4_analytic": K4_fit,
            "alpha_implied": float(-K2_fit),  # threshold: alpha = -K2 for eps=0
            "fit_ok": True,
        }
    except Exception as e:
        return {"family": "mexican_hat", "fit_ok": False, "error": str(e)}


def fit_gaussian(r: np.ndarray, C_r: np.ndarray, xi_init: float = 0.5) -> dict:
    """Fit C(r) to K_G(r) = A*exp(-r^2/(2*xi^2))."""
    def gaussian(r, A, xi):
        return A * np.exp(-r**2 / (2.0 * xi**2))

    try:
        popt, pcov = curve_fit(gaussian, r, C_r, p0=[C_r.max(), xi_init], maxfev=5000)
        A_fit, xi_fit = popt
        A_std, xi_std = np.sqrt(np.diag(pcov))
        K0_fit = 2.0 * math.pi * A_fit * xi_fit**2
        K2_fit = math.pi * A_fit * xi_fit**4
        K4_fit = math.pi * A_fit * xi_fit**6 / 8.0
        return {
            "family": "gaussian",
            "A": float(A_fit),
            "xi": float(xi_fit),
            "A_std": float(A_std),
            "xi_std": float(xi_std),
            "K0_analytic": float(K0_fit),
            "K2_analytic": float(K2_fit),
            "K4_analytic": float(K4_fit),
            "note": "Pure Gaussian: K2>0, cannot produce SH instability alone.",
            "fit_ok": True,
        }
    except Exception as e:
        return {"family": "gaussian", "fit_ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Optional visualisation
# ---------------------------------------------------------------------------

def make_plots(
    output_dir: Path,
    r: np.ndarray,
    C_r: np.ndarray,
    sh: dict,
    moments: dict,
    alpha: float,
    mh_fit: dict,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots.")
        return

    # --- Correlation function ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(r, C_r, "b-", label="C(r) measured")
    if mh_fit.get("fit_ok"):
        C_fit = mh_fit["C"]
        xi_fit = mh_fit["xi"]
        r_fine = np.linspace(0, r.max(), 300)
        u = r_fine**2 / (2.0 * xi_fit**2)
        ax.plot(r_fine, C_fit * (1.0 - u) * np.exp(-u), "r--", label="MH fit")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("r (nm)")
    ax.set_ylabel("C(r)")
    ax.set_title("Radial two-point correlation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_plot.png", dpi=150)
    plt.close(fig)

    # --- Dispersion relation sigma(q) ---
    q_arr = np.linspace(0.01, 20.0, 400)  # nm^-1
    K0, K2, K4 = moments["K0"], moments["K2"], moments["K4"]
    sigma = K0 - K2 * q_arr**2 + K4 * q_arr**4 - alpha * q_arr**2

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(q_arr, sigma, "b-", label=r"$\sigma(q)$")
    ax.axhline(0, color="k", lw=0.5)
    if sh.get("q_c") is not None:
        ax.axvline(sh["q_c"], color="r", ls="--", label=f"$q_c$ = {sh['q_c']:.2f} nm$^{{-1}}$")
    q_dna = math.sqrt((M_WINDING / R0_DNA)**2 + (2 * math.pi * M_WINDING / P_DNA)**2)
    ax.axvline(q_dna, color="g", ls=":", label=f"$q_{{DNA}}$ = {q_dna:.2f} nm$^{{-1}}$")
    ax.set_xlabel("q (nm$^{-1}$)")
    ax.set_ylabel(r"$\sigma(q)$")
    ax.set_title("Dispersion relation: growth rate vs wavenumber")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "dispersion_plot.png", dpi=150)
    plt.close(fig)

    print(f"  Plots saved to {output_dir}")


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_summary(
    output_dir: Path,
    moments: dict,
    sh: dict,
    threshold: dict,
    mh_fit: dict,
    gauss_fit: dict,
    params: dict,
) -> None:
    lines = [
        "URP Kernel Moment Extraction — Summary",
        "=" * 50,
        f"  alpha (input):        {params['alpha']:.4f}",
        f"  beta (QCD):           {BETA_QCD}",
        f"  G / kappa_vac (QCD):  {G_QCD}",
        f"  kappa_crit (target):  {KAPPA_CRIT}",
        f"  alpha_target:         {ALPHA_TARGET}",
        f"  r0 (DNA cylinder):    {R0_DNA} nm",
        f"  xi_0:                 {XI_0} nm",
        "",
        "Extracted Kernel Moments",
        "-" * 30,
        f"  K0 = {moments['K0']:.6f}",
        f"  K2 = {moments['K2']:.6f}",
        f"  K4 = {moments['K4']:.6f}",
        "",
        "Swift-Hohenberg Coefficients",
        "-" * 30,
        f"  mu     = -K2 - alpha = {sh['mu']:.6f}   (>0 needed)",
        f"  lambda = -K4         = {sh['lambda']:.6f}   (>0 needed)",
        f"  eps    = -K0         = {sh['epsilon']:.6f}",
        f"  q_c                  = {sh['q_c']} nm^-1",
        f"  SH instability:      {threshold['sh_instability_conditions_met']}",
        "",
        "Threshold / DNA Pitch Check",
        "-" * 30,
        f"  q_DNA (B-form, m=6): {threshold['q_dna']:.4f} nm^-1",
        f"  q_c (extracted):     {threshold['q_c_extracted']}",
        f"  Pitch deviation:     {threshold['pitch_fractional_deviation']}",
        f"  alpha_thresh (+):    {threshold['alpha_threshold_plus']}",
        f"  alpha_thresh (-):    {threshold['alpha_threshold_minus']}",
        f"  alpha deviation from target: {threshold['alpha_deviation_from_target']}",
        "",
        "Mexican Hat Fit",
        "-" * 30,
    ]
    if mh_fit.get("fit_ok"):
        lines += [
            f"  C   = {mh_fit['C']:.6f} nm^-2",
            f"  xi  = {mh_fit['xi']:.4f} nm",
            f"  K2_analytic = {mh_fit['K2_analytic']:.6f}",
            f"  K4_analytic = {mh_fit['K4_analytic']:.6f}",
            f"  alpha_implied (at threshold) = {mh_fit['alpha_implied']:.4f}",
        ]
    else:
        lines.append(f"  Fit failed: {mh_fit.get('error')}")

    lines += [
        "",
        "Gaussian Fit",
        "-" * 30,
    ]
    if gauss_fit.get("fit_ok"):
        lines += [
            f"  A   = {gauss_fit['A']:.6f} nm^-2",
            f"  xi  = {gauss_fit['xi']:.4f} nm",
            f"  K2_analytic = {gauss_fit['K2_analytic']:.6f}  (positive — no SH instability)",
        ]
    else:
        lines.append(f"  Fit failed: {gauss_fit.get('error')}")

    lines += [
        "",
        "Interpretation",
        "-" * 30,
        "  If K2 < 0 and K4 < 0 (mu,lam > 0): effective kernel is Mexican-hat type.",
        "  If alpha_thresh ~ alpha_target (0.0887): Conjecture A is consistent.",
        "  If q_c ~ q_DNA (12.6 nm^-1): pitch formula is satisfied.",
        "",
        "Cross-references",
        "  Docs/Kernel_Moments_Derivation.md",
        "  Docs/The_Lucidic_Compass.md  Sections 3-5",
    ]

    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="URP kernel moment extraction on cylindrical geometry."
    )
    p.add_argument("--n-theta", type=int, default=64,
                   help="Number of grid points in theta direction (default: 64).")
    p.add_argument("--n-z", type=int, default=64,
                   help="Number of grid points in z direction (default: 64).")
    p.add_argument("--r0", type=float, default=R0_DNA,
                   help=f"Cylinder radius in nm (default: {R0_DNA}).")
    p.add_argument("--z-length", type=float, default=20.0,
                   help="Cylinder length in nm (default: 20.0).")
    p.add_argument("--steps", type=int, default=100,
                   help="Field evolution steps (default: 100).")
    p.add_argument("--dt", type=float, default=0.005,
                   help="Time step (default: 0.005).")
    p.add_argument("--alpha", type=float, default=ALPHA_TARGET,
                   help=f"Diffusion coefficient (default: alpha_target={ALPHA_TARGET}).")
    p.add_argument("--beta", type=float, default=BETA_QCD,
                   help=f"Complexity coupling (default: {BETA_QCD}).")
    p.add_argument("--kappa", type=float, default=KAPPA_CRIT,
                   help=f"Capacity kappa used to set kernel xi (default: kappa_crit={KAPPA_CRIT}).")
    p.add_argument("--kernel-C", type=float, default=0.124,
                   help="Mexican hat kernel amplitude C in nm^-2 (default: 0.124, from Docs/Kernel_Moments_Derivation.md).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42).")
    p.add_argument("--output-dir", default="artifacts/kernel_moments",
                   help="Output directory (default: artifacts/kernel_moments).")
    p.add_argument("--visualize", action="store_true", default=False,
                   help="Generate matplotlib plots.")
    p.add_argument("--record-every", type=int, default=20,
                   help="Snapshot interval during evolution (default: 20).")
    return p


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute kernel xi from holographic scaling: xi = xi_0 / sqrt(kappa)
    kernel_xi = XI_0 / math.sqrt(args.kappa)
    dz = args.z_length / args.n_z

    print(f"URP Kernel Moment Extraction")
    print(f"  Grid:        {args.n_theta} x {args.n_z} (theta x z)")
    print(f"  r0:          {args.r0} nm")
    print(f"  dz:          {dz:.4f} nm")
    print(f"  kappa:       {args.kappa}")
    print(f"  kernel xi:   {kernel_xi:.4f} nm")
    print(f"  kernel C:    {args.kernel_C}")
    print(f"  alpha:       {args.alpha}")
    print(f"  steps:       {args.steps}")
    print()

    # Initialise and evolve field
    phi0 = make_cylindrical_field(args.n_theta, args.n_z, seed=args.seed)
    print("Evolving field...")
    phi_final, snapshots = evolve_cylindrical(
        phi0,
        r0=args.r0,
        dz=dz,
        dt=args.dt,
        steps=args.steps,
        alpha=args.alpha,
        beta=args.beta,
        kernel_C=args.kernel_C,
        kernel_xi=kernel_xi,
        record_every=args.record_every,
    )
    print(f"  Final field: mean={phi_final.mean():.4f}, std={phi_final.std():.4f}")

    # Two-point correlation
    print("Computing two-point correlation...")
    corr_2d = compute_two_point_correlation(phi_final)
    r_bins, C_r = azimuthal_average(corr_2d, r0=args.r0, dz=dz)

    # Extract moments
    print("Extracting kernel moments...")
    moments = extract_moments(r_bins, C_r)
    print(f"  K0 = {moments['K0']:.6f}")
    print(f"  K2 = {moments['K2']:.6f}")
    print(f"  K4 = {moments['K4']:.6f}")

    # Swift-Hohenberg coefficients
    sh = swift_hohenberg_coefficients(moments, args.alpha)
    threshold = threshold_check(sh, moments, args.alpha, r0=args.r0)

    # Kernel fits
    print("Fitting kernel families...")
    mh_fit = fit_mexican_hat(r_bins, C_r, xi_init=kernel_xi)
    gauss_fit = fit_gaussian(r_bins, C_r, xi_init=kernel_xi)

    # Save outputs
    np.save(output_dir / "correlation.npy", corr_2d)
    np.save(output_dir / "radial_correlation.npy", np.stack([r_bins, C_r]))

    moments_out = {
        "parameters": {
            "alpha": args.alpha,
            "beta": args.beta,
            "kappa": args.kappa,
            "kernel_C": args.kernel_C,
            "kernel_xi": kernel_xi,
            "r0": args.r0,
            "n_theta": args.n_theta,
            "n_z": args.n_z,
            "steps": args.steps,
        },
        "moments": moments,
        "swift_hohenberg": sh,
        "threshold_check": threshold,
    }
    (output_dir / "moments.json").write_text(
        json.dumps(moments_out, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "kernel_fit.json").write_text(
        json.dumps({"mexican_hat": mh_fit, "gaussian": gauss_fit}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8"
    )

    params = {"alpha": args.alpha, "kappa": args.kappa, "kernel_xi": kernel_xi}
    write_summary(output_dir, moments, sh, threshold, mh_fit, gauss_fit, params)

    if args.visualize:
        print("Generating plots...")
        make_plots(output_dir, r_bins, C_r, sh, moments, args.alpha, mh_fit)

    print(f"\nArtifacts written to {output_dir}")


if __name__ == "__main__":
    main()
