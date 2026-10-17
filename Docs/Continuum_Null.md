# The continuum null: `d+1` in any dimension, with no lattice

### The geometric half of the junction selection, derived for all `d` instead of simulated up to `d = 4`

*This note stands alone. It assumes no framework. It extends one half of
`Junction_Selection.md` — the null model — past the reach of any box the
programme can afford. Status tags: **[theorem]** — a standard result, cited,
not re-proved; **[derived]** — follows here from stated assumptions;
**[argument]** — reasoning that is not yet a proof; **[spot-check]** —
verified numerically this session, with the repo instrument pre-registered
below but not yet run; **[post hoc]** — read against already-published
numbers after the fact.*

---

## The claim

Take a Poisson–Voronoi tessellation of R^d (intensity λ), colour every cell
independently and uniformly from a palette of `P` sectors, and interrogate
each vertex with a probe ball of radius `ε`, small against the typical cell
scale `ℓ = λ^(−1/d)`. Write `η = ε/ℓ`. Ask of each vertex: *does the probe
ball contain all `P` colours?*

The full-palette vertex intensity is:

| palette | intensity | standing |
|---|---|---|
| `P ≤ d` | exactly `0` | **[derived]** — counting |
| `P = d+1` | `λ_v · (d+1)!/(d+1)^(d+1) · (1 − O(η))` | **[derived]** |
| `P = d+k`, `k ≥ 2` | `λ_v · C_{d,k} · η^(k−1) · (1 + o(1))` | **[derived]** to leading order; constants not computed |

where `λ_v` is the vertex intensity (`2λ` in 2-D). So in **every** dimension
the argmax over `P` is exactly `d+1`, and the margin over the nearest rival
diverges like `η^(−1)` as the probe sharpens. No lattice, no box, no
relaxation. The `d = 5` ceiling recorded in `Junction_Selection.md` was
instrumental, and for the null half of the claim it is gone.

## Why: three ingredients

**1. Simplicity [theorem].** Almost surely, every vertex of a Poisson–Voronoi
tessellation in R^d has exactly `d+1` incident cells: `d+2` co-spherical
Poisson points are a measure-zero event (Miles 1970; Møller, *Lectures on
Random Voronoi Tessellations*). This is precisely the "generic vertex" the
lattice programme checks for empirically; here it is a property of the model.

**2. The vertex is interior to the union of its incident cells [theorem].**
Voronoi tessellations are normal: a vertex lies in the interior of the union
of its `d+1` closed cells. So for each vertex `v` there is a clearance
`g(v) > 0` — the distance from `v` to the nearest non-incident cell — and a
probe ball with `ε < g(v)` sees exactly the incident colours.

**3. The clearance distribution has positive density at zero [derived].**
Condition on the empty circumsphere (radius `R`, `d+1` boundary sites). By
Slivnyak–Mecke the outside process is still Poisson(λ), so the gap
`δ = min|v−x| − R` to the nearest outside site is exponential with rate
`λ·S(R)`, where `S(R) = d·ω_d·R^(d−1)` is the sphere's surface area. The
clearance is the gap divided by an angular factor that is `O(1)` almost
surely, so

```
    P(g < ε)  ~  λ · E[S(R)·a(θ)] · ε   +   O(ε²)
```

— **linear in `ε`, in every dimension.** This is the load-bearing order, and
the natural guess gets it wrong. One might expect `η^d` ("another vertex
falls within `ε` of ours" — a `d`-dimensional rarity). The binding event is
not that: it is one incident edge being short, or one outside site nearly
touching the circumsphere — a **one-dimensional** rarity, because the gap is
a single radial variable with a smooth density at zero.

The colouring is then pure combinatorics. `P ≤ d`: a vertex carries at most
`d+1` colours, so full palette plus `d+1` distinct is impossible — zero by
counting. `P = d+1`: the `d+1` incident cells must all differ,
probability `(d+1)!/(d+1)^(d+1)`. `P = d+k`: the ball must catch at least
`k−1` non-incident cells, costing `η^(k−1)`, times a colour factor that is
`O(1)` and irrelevant to the order.

## Spot-checks run this session **[spot-check]**

Box-simulated Poisson–Voronoi, periodic tiling, unit intensity:

| check | prediction | measured |
|---|---|---|
| simplicity, d = 2 (1000 vertices) | every vertex 3 cells | 1000/1000 |
| simplicity, d = 3 (2708 vertices) | every vertex 4 cells | 2708/2708 |
| colour constant, d = 2 | `3!/3³ = 0.2222` | `0.212` (sampling σ ≈ 0.013) |
| colour constant, d = 3 | `4!/4⁴ = 0.0938` | `0.090` (sampling σ ≈ 0.006) |
| clearance law, d = 2 (4000 vertices) | `P(g<ε)` linear at small `ε` | ratio `2.0` per doubling of `ε` at the low end; bends over only at the gap scale, as the exponential CDF must |

These are in-session checks, not the repo instrument. The pre-registered
instrument below is what makes them reproducible here.

## Consistency with the published lattice null **[post hoc]**

`experiments/n3_junction_null.py` measured non-zero `P = d+2` readings and
logged them as "lattice artefacts: a `3^d` window can straddle two nearby
generic vertices." The continuum null says what that artefact *is*:
the `η¹` term. Read the three published arms as probes at fixed
`ε ≈ 1.5` lattice units and varying `ℓ`:

| arm | `η = ε/ℓ` | measured `P(d+2)/P(d+1)` | implied `c_d` = ratio/η |
|---|---|---|---|
| 2-D, 160², 70 seeds | `0.079` | `0.0285` | `0.36` |
| 3-D, 56³, 40 seeds | `0.091` | `0.053` | `0.58` |
| 4-D, 26⁴, 24 seeds | `0.128` | `0.22` | `1.7` |

Under the `η¹` law the implied constant is the same order in all three arms
and grows mildly — as it should, since the `3^d` window's surface through
which extra cells can enter grows with `d`. Under the rejected `η^d` law the
same numbers would require `c'_d ≈ 5, 70, 800` — three orders of magnitude of
unmotivated fine-tuning. The published data were already consistent with the
derivation; they just hadn't been read that way. The relaxed-field radius
sweep in `Junction_Selection.md` (margin `∞ → 45.5 → 16.6` as the probe
widens `1 → 2 → 3`) is qualitatively the same statement: the margin falls as
the `η¹` term grows.

## What this establishes, and what it does not

**It establishes** that the geometric half of the selection — the half the
Voronoi null already showed needs no dynamics — is dimension-independent. The
selection at `d+1` is not a property of small boxes in low dimensions; it is
what any generic i.i.d.-coloured tessellation does in any `d`, with a
margin the probe controls.

**It does not** say anything new about relaxed fields. Their role in the
programme's current reading is genericity — degenerate vertices are unstable
under the dynamics — and that remains a separate, measured claim
(`n3_junction_genericity.py`). Three caveats stay open:

- **Real fields have correlated colours.** Volume conservation correlates
  neighbouring cells' sectors. The *order structure* (`0` / constant /
  `η^(k−1)`) survives any colouring short of rigid correlation
  **[argument]**; the `P = d+1` constant `(d+1)!/(d+1)^(d+1)` assumes i.i.d.
  and will move. Measuring how much it moves on relaxed fields is the
  interesting version of the instrument.
- **The constants `C_{d,k}` are uncomputed.** The instrument measures them;
  the derivation fixes only the exponent.
- **Poisson–Voronoi is maximally disordered.** Equilibrium tessellations are
  sub-Poisson. Simplicity is robust to that (it is a general-position
  property); the clearance law's *rate* is not.

Nothing here touches the capacity half of the programme.

## What would refute it

- The instrument measuring a `P = d+2` exponent other than `1` as `η → 0`, in
  any dimension.
- The instrument measuring a `P = d+1` constant other than
  `(d+1)!/(d+1)^(d+1)` under i.i.d. colouring, beyond sampling error.
- A relaxed field, confirmed generic on both instruments, whose full-palette
  fraction deviates from the null prediction at matched colouring
  statistics. That would mean the dynamics do selection work beyond
  genericity — which the programme's current reading denies, and which would
  be the most interesting possible outcome.

## The instrument (pre-registered, not yet run)

`experiments/n3_continuum_null.py` — a typical-vertex Palm Monte Carlo via
the empty-ball construction: sample the circumradius from its known density
(`f(r) ∝ r^(d²−1) e^(−λω_d r^d)`), place `d+1` directions i.i.d. uniform on
the sphere weighted by simplex volume, generate the outside gap from its
exponential law, and colour. Pure numpy/numba — no scipy, per repo
constraint — and `O(1)` per vertex in any dimension, so `d = 2…6` costs
minutes. Box-simulation arms in `d = 2, 3` pin it against the spot-checks
above.

Predictions, registered before the first run:

- **P1.** The `P = d+1` full-palette fraction equals `(d+1)!/(d+1)^(d+1)`
  within 3σ of sampling error, for `d = 2, 3, 4, 5`.
  *Falsifier: any arm outside 3σ.*
- **P2.** The `P = d+2` fraction scales as `η^a` with fitted `a` within
  `±0.15` of `1` over `η ∈ [0.005, 0.05]`, for `d = 2, 3, 4`.
  *Falsifier: any arm outside the band.*
- **P3.** The argmax over `P ∈ {2,…,8}` is `d+1` at every `η ≤ 0.1` in every
  dimension run. *Falsifier: any other argmax.*
- **P4.** The measured `c_d` reproduce the three published lattice arms'
  ratios within a factor of 2 (the window geometries differ, so only the
  order can be demanded). *Falsifier: any arm off by more.*

---

*The lattice programme found that `d+1` is selected and then found it was
geometry doing the selecting. This note is that second finding with the
scaffolding removed: the selection is what genericity and counting already
imply, in any dimension, and the only thing a simulation was ever adding was
the noise floor.*
