"""Three-component sector field — the Ψ∈ℂ³ layer of the URP gauge picture.

The scalar URP field (see :mod:`project_genesis.sectorisation`) can only form
*layered* domains: a region in well ``n`` borders wells ``n±1``, so three
mutually-adjacent phases — and the 120° Y-junctions the gauge derivation ties
to colour SU(3) — cannot arise from a single scalar.

This module implements the next layer the gauge paper describes (§4.3.3): a
*sector-membership field* ``Ψ(x) = (R, G, B)`` whose three components compete
on equal footing. Deep inside a sector one component dominates (Ψ ≈ a basis
vector); near boundaries the components mix. With all three phases mutually
adjacent, genuine three-way domains with 120° triple junctions form.

Dynamics are vector Allen–Cahn (overdamped gradient descent on a triple-well
free energy), the natural multi-phase generalisation of the scalar URP update:

    ∂_t η_a = D·∇²η_a − ∂f/∂η_a,
    f(η)    = Σ_a (¼η_a⁴ − ½η_a²) + γ·Σ_{a<b} η_a²·η_b²,
    ∂f/∂η_a = η_a³ − η_a + 2γ·η_a·(Σ_b η_b² − η_a²).

The free energy is S₃-symmetric (permuting R/G/B is a symmetry), the discrete
analogue of the global relabelling symmetry that survives deep inside sectors.
Implemented with vectorised periodic NumPy so it is dimension-agnostic (2-D for
the browser toy, 3-D to match the engine).
"""

from __future__ import annotations

import numpy as np


def periodic_laplacian(field: np.ndarray) -> np.ndarray:
    """Discrete Laplacian with periodic boundaries, for any dimensionality."""
    out = -2.0 * field.ndim * field
    for axis in range(field.ndim):
        out = out + np.roll(field, 1, axis=axis) + np.roll(field, -1, axis=axis)
    return out


def step_multiphase(
    fields: np.ndarray,
    *,
    diffusion: float = 1.0,
    gamma: float = 1.5,
    dt: float = 0.1,
) -> np.ndarray:
    """Advance a multi-phase sector field one Allen–Cahn step.

    Parameters
    ----------
    fields:
        Array of shape ``(P, *spatial)`` holding the ``P`` competing
        components (``P = 3`` for the R/G/B colour sectors).
    diffusion:
        Gradient (surface-tension) coefficient ``D``.
    gamma:
        Cross-coupling penalty. ``γ > 1`` makes the three unit-corner states
        ``(1,0,0)``, ``(0,1,0)``, ``(0,0,1)`` the degenerate minima, so each
        cell is driven toward a single dominant phase.
    dt:
        Explicit time step.

    Returns
    -------
    The updated ``(P, *spatial)`` field array.
    """
    if fields.ndim < 2:
        raise ValueError("fields must have shape (P, *spatial) with P >= 1 component axis")
    sum_sq = np.sum(fields * fields, axis=0)
    new = np.empty_like(fields)
    for a in range(fields.shape[0]):
        fa = fields[a]
        lap = periodic_laplacian(fa)
        df = fa**3 - fa + 2.0 * gamma * fa * (sum_sq - fa * fa)
        new[a] = fa + dt * (diffusion * lap - df)
    return new


def _total_gradient_energy(fields: np.ndarray) -> np.ndarray:
    """Σ_a |∇η_a|² per voxel — the total distinction load across components."""
    load = np.zeros(fields.shape[1:], dtype=np.float64)
    for a in range(fields.shape[0]):
        for axis in range(fields[a].ndim):
            g = 0.5 * (np.roll(fields[a], -1, axis) - np.roll(fields[a], 1, axis))
            load += g * g
    return load


def step_multiphase_conserved(
    fields: np.ndarray,
    kappa: np.ndarray | None = None,
    *,
    diffusion: float = 1.0,
    gamma: float = 1.5,
    dt: float = 0.1,
    kappa_baseline: float = 1.0,
    kappa_recovery: float = 0.1,
    kappa_consumption: float = 5.0,
    kappa_diffusion: float = 0.5,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Advance a multi-phase field with **volume-conserving** (junction-resolving)
    dynamics.

    Plain Allen–Cahn coarsens without bound — every junction is transient and
    the field collapses to a single domain. Conserving each component's total
    (a global Lagrange multiplier: subtract the spatial mean of the bulk drift
    per component) fixes the phase fractions, so the system cannot eliminate a
    phase. It settles into a *stable* multi-domain tiling whose 120° triple
    junctions persist — the structure the topological integration measure needs.

    If ``kappa`` is supplied it co-evolves and gates the diffusion exactly as in
    :func:`step_multiphase_kappa`. Returns ``(fields, kappa)`` (``kappa`` is
    ``None`` when not used).
    """
    if fields.ndim < 2:
        raise ValueError("fields must have shape (P, *spatial) with P >= 1 component axis")
    if kappa is not None and kappa.shape != fields.shape[1:]:
        raise ValueError("kappa must have the spatial shape of the field")
    sum_sq = np.sum(fields * fields, axis=0)
    new = np.empty_like(fields)
    load = np.zeros(fields.shape[1:], dtype=np.float64)
    # Spatial diffusion coefficient: κ-gated where κ is active, else uniform.
    diff_coeff = diffusion if kappa is None else kappa * diffusion

    for a in range(fields.shape[0]):
        fa = fields[a]
        lap = periodic_laplacian(fa)
        df = fa**3 - fa + 2.0 * gamma * fa * (sum_sq - fa * fa)
        df_conserved = df - df.mean()  # subtract mean => ∫η_a conserved
        new[a] = fa + dt * (diff_coeff * lap - df_conserved)
        for ax in range(fa.ndim):
            g = 0.5 * (np.roll(fa, -1, ax) - np.roll(fa, 1, ax))
            load += g * g

    if kappa is None:
        return new, None

    lap_k = periodic_laplacian(kappa)
    new_kappa = kappa + dt * (
        kappa_diffusion * lap_k
        + kappa_recovery * (kappa_baseline - kappa)
        - kappa_consumption * load * kappa
    )
    np.clip(new_kappa, 0.0, kappa_baseline, out=new_kappa)
    return new, new_kappa


def step_multiphase_kappa(
    fields: np.ndarray,
    kappa: np.ndarray,
    *,
    diffusion: float = 1.0,
    gamma: float = 1.5,
    dt: float = 0.1,
    kappa_baseline: float = 1.0,
    kappa_recovery: float = 0.1,
    kappa_consumption: float = 5.0,
    kappa_diffusion: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance a multi-phase field with a co-evolving capacity field κ.

    This is the three-component (Ψ∈ℂ³) analogue of the scalar engine's
    dynamical-κ coupling: capacity gates the *integrating* (surface-tension)
    term while the bulk triple-well potential is untouched, and κ is consumed
    by the total distinction load across all components:

        ∂_t η_a = κ·D·∇²η_a − ∂f/∂η_a
        ∂_t κ   = D_κ·∇²κ + r·(κ₀ − κ) − c·(Σ_b |∇η_b|²)·κ

    Returns the updated ``(fields, kappa)`` pair. Where capacity is depleted
    (at the dense domain walls), integration stalls and the wall network is
    pinned — so scarcity can arrest the otherwise unbounded coarsening at a
    finite number of genuine three-way domains.
    """
    if fields.ndim < 2:
        raise ValueError("fields must have shape (P, *spatial) with P >= 1 component axis")
    if kappa.shape != fields.shape[1:]:
        raise ValueError("kappa must have the spatial shape of the field")

    sum_sq = np.sum(fields * fields, axis=0)
    new = np.empty_like(fields)
    load = np.zeros(fields.shape[1:], dtype=np.float64)
    for a in range(fields.shape[0]):
        fa = fields[a]
        lap = periodic_laplacian(fa)
        df = fa**3 - fa + 2.0 * gamma * fa * (sum_sq - fa * fa)
        new[a] = fa + dt * (kappa * diffusion * lap - df)
        g_axes = [0.5 * (np.roll(fa, -1, ax) - np.roll(fa, 1, ax)) for ax in range(fa.ndim)]
        load += sum(g * g for g in g_axes)

    lap_k = periodic_laplacian(kappa)
    new_kappa = kappa + dt * (
        kappa_diffusion * lap_k
        + kappa_recovery * (kappa_baseline - kappa)
        - kappa_consumption * load * kappa
    )
    np.clip(new_kappa, 0.0, kappa_baseline, out=new_kappa)
    return new, new_kappa


def multiphase_s_functional(
    fields: np.ndarray,
    prev_fields: np.ndarray | None,
    *,
    beta: float = 0.09,
    kappa_field: np.ndarray | None = None,
) -> dict[str, float]:
    """S = ΔC + κΔI for a multi-phase field.

    Distinction is the total gradient energy across components; integration is
    the step-over-step reduction in mean component curvature (smoothing). When
    a dynamical ``kappa_field`` is supplied its mean is used for κ, otherwise
    the diagnostic proxy ``1/(1+mean load)``.
    """
    load = _total_gradient_energy(fields)
    mean_load = float(load.mean())
    delta_c = float(beta * mean_load)
    if kappa_field is not None:
        kappa = float(kappa_field.mean())
    else:
        kappa = 1.0 / (1.0 + mean_load)

    if prev_fields is None:
        delta_i = 0.0
    else:
        def mean_abs_lap(f: np.ndarray) -> float:
            return float(np.mean([np.abs(periodic_laplacian(f[a])) for a in range(f.shape[0])]))
        delta_i = max(0.0, mean_abs_lap(prev_fields) - mean_abs_lap(fields))

    return {
        "delta_c": delta_c,
        "kappa": kappa,
        "delta_i": delta_i,
        "s_increment": delta_c + kappa * delta_i,
    }


def count_domains(labels: np.ndarray, *, min_size: int = 1) -> int:
    """Count connected single-phase domains (periodic, 6-connectivity).

    Unlike :func:`count_triple_junctions` (which counts meeting points), this
    counts the regions themselves — the emergent ``N`` the phase-diagram work
    needs in order to ask whether κ scarcity stabilises *three* domains.
    """
    try:
        from scipy import ndimage
    except ImportError:  # pragma: no cover
        raise RuntimeError("scipy is required for count_domains")
    from .sectorisation import _merge_periodic_faces

    structure = ndimage.generate_binary_structure(labels.ndim, 1)
    total = 0
    for val in np.unique(labels):
        mask = labels == val
        comp, n = ndimage.label(mask, structure=structure)
        if n > 1:
            comp = _merge_periodic_faces(comp, n)
            n = int(comp.max())
        if min_size > 1 and n > 0:
            counts = np.bincount(comp.ravel(), minlength=n + 1)
            n = int(np.sum(counts[1:] >= min_size))
        total += n
    return total


def coherence_integration(
    fields: np.ndarray,
    *,
    radius: int = 3,
    decay: float = 1.0,
) -> float:
    """Standing nonlocal coherence I = Σ_a ⟨η_a(x)·η_a(x+δ)⟩ weighted by exp(−decay·|δ|).

    The multi-component analogue of the URP integration functional
    ``I[φ] = ∫∫ K(x,x') φ(x) φ(x')``. Unlike the transient ΔI (a one-step
    smoothing rate that vanishes at equilibrium), this is a *standing* quantity:
    a static coherent domain keeps it high, while fragmentation lowers it
    because domain walls decorrelate neighbourhoods within the kernel radius.
    It therefore survives at equilibrium and falls monotonically with sector
    count — the integration half the S-functional was missing.

    Normalised by the kernel-weight sum so the value is intensive (independent
    of radius/decay scale).
    """
    from itertools import product

    spatial = fields.shape[1:]
    ndim = len(spatial)
    coherence = 0.0
    weight_sum = 0.0
    for off in product(range(-radius, radius + 1), repeat=ndim):
        dist = float(np.sqrt(sum(o * o for o in off)))
        if dist == 0.0 or dist > radius:
            continue
        w = float(np.exp(-decay * dist))
        weight_sum += w
        for a in range(fields.shape[0]):
            shifted = fields[a]
            for axis, o in enumerate(off):
                if o:
                    shifted = np.roll(shifted, -o, axis=axis)
            coherence += w * float(np.mean(fields[a] * shifted))
    return coherence / weight_sum if weight_sum else 0.0


def multiphase_s_standing(
    fields: np.ndarray,
    *,
    beta: float = 0.09,
    kappa_field: np.ndarray | None = None,
    radius: int = 3,
    decay: float = 1.0,
    integration_weight: float = 1.0,
) -> dict[str, float]:
    """S = ΔC + κ·w·I using the *standing* nonlocal coherence for integration.

    Distinction ΔC rises with fragmentation (more walls); the coherence term
    falls with it — so unlike :func:`multiphase_s_functional`, this S can have
    an interior optimum in sector count. ``integration_weight`` sets the
    relative scale of the two halves (the b/a ratio of the F(N) tradeoff).
    """
    load = _total_gradient_energy(fields)
    mean_load = float(load.mean())
    delta_c = float(beta * mean_load)
    kappa = float(kappa_field.mean()) if kappa_field is not None else 1.0 / (1.0 + mean_load)
    coherence = coherence_integration(fields, radius=radius, decay=decay)
    return {
        "delta_c": delta_c,
        "kappa": kappa,
        "coherence": coherence,
        "s_increment": delta_c + kappa * integration_weight * coherence,
    }


def sector_labels(fields: np.ndarray) -> np.ndarray:
    """Return the dominant-component (argmax) sector label at each voxel."""
    return np.argmax(fields, axis=0)


def interface_mask(fields: np.ndarray, *, margin: float = 0.15) -> np.ndarray:
    """Boolean mask of boundary cells where the top two components are close.

    A cell is on a domain wall when no single phase clearly dominates — i.e.
    the gap between the largest and second-largest component is below
    ``margin``.
    """
    srt = np.sort(fields, axis=0)
    gap = srt[-1] - srt[-2]
    return gap < margin


def _neighbourhood_label_bits(labels: np.ndarray) -> np.ndarray:
    """Per-cell bit-set of the sector labels present in its 3^d neighbourhood."""
    bits = 1 << labels  # include the cell's own label
    for off in _neighbour_offsets(labels.ndim):
        shifted = labels
        for axis, delta in enumerate(off):
            if delta:
                shifted = np.roll(shifted, -delta, axis=axis)
        bits = bits | (1 << shifted)
    return bits


def count_triple_junctions(labels: np.ndarray) -> int:
    """Count cells whose local neighbourhood contains three or more sectors.

    For a multi-phase field these are genuine triple points — the discrete
    analogue of the 120° Y-junctions where three colour domains meet. Works in
    2-D and 3-D via periodic neighbour shifts.
    """
    return int(np.sum(_popcount(_neighbourhood_label_bits(labels)) >= 3))


def full_palette_junction_density(labels: np.ndarray, n_palette: int) -> float:
    """Density of junctions that represent the *complete* colour palette.

    A cell qualifies when its neighbourhood contains a genuine junction
    (≥ 3 distinct sectors) *and* all ``n_palette`` colours at once. This is the
    discrete form of the gauge paper's §6 neutrality criterion: a junction is
    colour-neutral only when it carries the whole palette.

    The geometry does the selecting. In 2-D, junctions are generically 3-fold
    (three domains meet at a point), so a junction can show *all* the colours
    only when the palette is exactly three: a 2-colour system forms no triple
    junctions, and a ≥ 4-colour system cannot fit its whole palette onto a
    3-fold vertex. The measure is therefore non-zero essentially only at
    ``n_palette = 3`` — the non-collinear, topological selector the coherence
    magnitude lacked. (This is a 2-D structural argument; 3-D junction lines
    have different valence and the clean selection need not transfer.)
    """
    bits = _neighbourhood_label_bits(labels)
    full = (1 << n_palette) - 1
    distinct = _popcount(bits)
    return float(np.mean((bits == full) & (distinct >= 3)))


def topological_s_functional(
    fields: np.ndarray,
    *,
    n_palette: int | None = None,
    beta: float = 0.09,
    kappa_field: np.ndarray | None = None,
    weight: float = 1.0,
) -> dict[str, float]:
    """S = ΔC + κ·w·(full-palette junction density).

    Unlike the coherence-magnitude integration (which is collinear with ΔC),
    the neutrality term is non-zero essentially only for a three-colour palette,
    so this S has an interior optimum at three sectors. ``n_palette`` defaults
    to the number of components.
    """
    n = fields.shape[0] if n_palette is None else n_palette
    load = _total_gradient_energy(fields)
    mean_load = float(load.mean())
    delta_c = float(beta * mean_load)
    kappa = float(kappa_field.mean()) if kappa_field is not None else 1.0 / (1.0 + mean_load)
    neutrality = full_palette_junction_density(sector_labels(fields), n)
    return {
        "delta_c": delta_c,
        "kappa": kappa,
        "neutrality": neutrality,
        "s_increment": delta_c + kappa * weight * neutrality,
    }


def _neighbour_offsets(ndim: int) -> list[tuple[int, ...]]:
    """All non-zero offsets in the 3^ndim neighbourhood."""
    from itertools import product

    offsets = [o for o in product((-1, 0, 1), repeat=ndim) if any(o)]
    return offsets


def _popcount(arr: np.ndarray) -> np.ndarray:
    """Vectorised population count for a small-int bit-set array."""
    out = np.zeros_like(arr)
    a = arr.copy()
    while np.any(a):
        out += a & 1
        a >>= 1
    return out


def analyze_multiphase(
    fields: np.ndarray,
    *,
    beta: float = 0.09,
    kappa_field: np.ndarray | None = None,
    count_domains_too: bool = True,
) -> dict:
    """Summary report for a multi-phase sector field.

    Reports the number of distinct phases present, the emergent domain count,
    the triple-junction count (genuinely non-zero, unlike the scalar model),
    the boundary fraction, and a distinction/integration decomposition. When a
    ``kappa_field`` is supplied, its mean and wall/interior split are included.
    """
    labels = sector_labels(fields)
    walls = interface_mask(fields)
    mean_grad = float(_total_gradient_energy(fields).mean())
    report = {
        "n_phases": int(len(np.unique(labels))),
        "triple_junctions": count_triple_junctions(labels),
        "boundary_fraction": float(walls.mean()),
        "distinction": float(beta * mean_grad),
        "integration": float(1.0 / (1.0 + mean_grad)),
    }
    if count_domains_too:
        report["n_domains"] = count_domains(labels)
    if kappa_field is not None:
        report["kappa_mean"] = float(kappa_field.mean())
        report["wall_kappa_mean"] = float(kappa_field[walls].mean()) if walls.any() else 0.0
        report["interior_kappa_mean"] = (
            float(kappa_field[~walls].mean()) if (~walls).any() else 0.0
        )
    return report
