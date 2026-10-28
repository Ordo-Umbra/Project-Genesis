# What a field with no URP in it scores

### Each measured claim, read against a baseline with no capacity, no S and no dynamics

*Code: `experiments/n3_null_audit.py`, `tests/test_null_audit.py`. Predictions
were committed (`d02d961`) before any arm ran. Run date 2026-10-28.*

---

## The short version

Every claim tested here that `The_Principle.md` credits to URP is reproduced
by a baseline with no URP in it: pure noise, a random Voronoi tessellation, a
Gaussian random field, or random curves. All 8 pre-registered predictions held.

That does not make the claims false. It means they are **consistent with**
URP but are **not evidence for it** over any other account that produces a
local field. A principle claimed to be universal should be satisfied by
generic fields; the cost is that those satisfactions cannot tell it apart from
no principle at all.

What is left as candidate evidence specific to URP is a short list of results
that depend on κ's own dynamics. None of them has yet been compared against a
mechanism without κ. That list is at the end.

## The audit

| claim | where | baseline | result | standing |
|---|---|---|---|---|
| `d + 1` selection | §3 | Voronoi | selects `d + 1` in d = 2, 3, 4 (`n3_junction_null`) | reproduced |
| The cliff: φ = 1 at P = 3, collapses above | §2 | noise, Voronoi | φ(3) = 1 on **pure noise**; φ(P ≥ 4) = 0.008 on Voronoi | identity + geometry |
| The gap: n_C − n_I = 1, capacity-invariant | §2 | Gaussian, Voronoi | field 1.04; Voronoi 1.04–1.06; Gaussian 0.98–1.01 | reproduced |
| Survives widening the probe | §3 | Voronoi | argmax 3 at radius 1, 2, 3 | ranking reproduced |
| Eviction toggle on three substrates | §5 | random curves | 100% evicted driven, 0% free, 100% agreement | guaranteed by the formula |
| Joint criterion selects three, 88× | §6 | §6's own table | churn varies 2.6× across P; integration carries it | reduces to the cliff |
| Spin, statistics, ½-defects | §3 | — | textbook nematic topology; κ enters only as an imposed pinning well | by inspection |
| κ-gravity: screened well, `V ∝ m₁m₂`, `c = √(D/τ)` | §4 | — | follows from a linear screened (telegraph) equation | by inspection |
| Dark energy from `r(κ₀−κ)` | §4 | — | a term that does not dilute behaves as Λ | by construction |
| Euler `2:3:1`, `d + 1` families | §3 | — | Euler's formula on any trivalent tiling | by inspection |

## The four arms, in brief

**A1, the cliff.** `φ = I/C` is the share of cells seeing 3+ sectors that see
the whole palette. At `P = 3`, a cell that sees 3 sectors sees all of them, so
`φ = 1` for **any** labelling: pure random noise reads `1.000000` in 2-D and
3-D. Above three, noise reads `0.72`, but any tessellation collapses it (Voronoi
`0.008`). The cliff is a counting identity followed by generic geometry. The
"capacity sweep cannot move φ" finding follows: an identity cannot move.

**A2, the gap.** Same instrument, defaults and calibration as `n3_area_law`.
The bias calibration reproduces (`+0.119` against the published `+0.12`), and
so does the field (`gap = 1.042`, published `1.03–1.04`).

| field | `n_C` | `n_I` | gap | raw gap |
|---|---|---|---|---|
| published field, c = 1 | 1.995 | 0.953 | **1.042** | 0.923 |
| Voronoi, width 6 / 10 / 16, σ = 1 | 1.98–2.02 | 0.95–0.98 | **1.035–1.060** | 0.916–0.942 |
| Voronoi, width 10, σ = 2 | 2.002 | 0.968 | **1.035** | 0.916 |
| Gaussian, ξ = 1.5 / 3 / 6 | 1.96–2.00 | 0.97–1.02 | **0.979–1.007** | 0.861–0.888 |

The field sits **inside** the Voronoi range. The mechanism was stated in
advance: distinction sums a stationary density over an area, so it scales
with the area; integration is computed from the two-point correlation across
the region's edge, so it scales with the edge once the region is wider than
the correlation length. That holds for any stationary, short-range field. The
gap measures the instrument's geometry, not the field.

**A3, the radius sweep.** §3 calls this "the only claim in this programme to
come out of a convention sweep wider than it went in". On Voronoi, `P = 3`
still wins at every radius, so the survival is geometric.

| radius | Voronoi margin | relaxed margin (published) |
|---|---|---|
| 1 | 57.7 | ∞ |
| 2 | 20.6 | 45.5 |
| 3 | 10.5 | 16.6 |

*Post hoc, not pre-registered:* the relaxed field's margin is larger at every
radius. That is the one place in this audit where the dynamics show an edge
over the null. It fits `Junction_Selection.md`'s genericity finding (relaxation
splits degenerate junctions, so fewer rivals score). Two cautions: the relaxed
sweep used the defective published seeding, and three trials each. Also, the
dynamics doing it are plain multiphase Allen–Cahn, standard surface-tension
physics with no κ in it, so the edge is not specific to URP either.

**A4, the eviction condition.** The transplants apply the capacity law after
the run, to measured curves, so the verdict is a function of those curves
alone. On 2,000 random curves meeting the transplants' two design premises
(the ordered point starts as the S optimum, and it is not the ΔC peak):

| | evictable agrees with ladder | ladder evicts |
|---|---|---|
| driven (every load > 0) | 100% | **100%** |
| free (ordered load = 0) | 100% | **0%** |

This is a theorem, and the run only illustrates it. As consumption grows,
every driven κ falls to 0 and S falls to ΔC, whose peak is not the ordered
point. A free ordered point keeps κ = 1 while every rival falls. So "16/16
driven evictable, 0/16 undriven" is guaranteed on any substrate that meets
the premises. Reproducing it on three substrates is not evidence that they
share anything. What is empirical is whether each substrate **meets the
premises**, and §5 already found that the first premise nearly fails on the
oscillators (headroom `0.24×`).

## What is left

These claims depend on κ's own dynamics and have a baseline of some kind, but
none against a mechanism without κ:

| claim | baseline it has | what's missing |
|---|---|---|
| κ-gated self-repair at `r = 0.8` (`Experiment_Log`, backfill) | the same field without κ fails | a non-URP regenerating resource; and `r` was set by hand |
| Continual learning: capacity-gated plasticity | **external**: beats SGD (`p < 0.001`), loses to replay | already tested against outside methods; the result is mixed |
| Memory stays connected below random `p_c` | random site percolation (uncorrelated) | a correlated null: any field with thin walls may do the same |

These are where URP could show something a generic field does not.

## Measured, consistent with, not spoken to

- **Measured:** the readings above, on the baselines named, at the sizes named.
- **Consistent with:** URP's claim that the asymmetry between distinction and
  integration is universal. Generic fields have it.
- **Not spoken to:** whether URP's dynamics are the *reason* any real system
  has these properties. Locality and generic geometry are already enough.

## Scope

2-D for A2 and A3; one consumption value for the field arm of A2; 48
realisations per A2 arm, 3 trials per palette in A1b and A3. The bias
correction in A2 is calibrated on the Gaussian control itself, so `n_I ≈ 1` on
that control is partly by construction; the Voronoi arms are the independent
ones. Voronoi colours are random, so neighbouring cells sometimes share a
colour and merge.

## What this changes, and what it does not

It does not change any measurement in the repo. It changes what several of
them are evidence **for**. `README.md` and `The_Principle.md` §2, §3, §5 and
§6 present these readings as support for URP specifically; they have not been
edited here. The README's proposed next experiment, testing whether the
formal half's flat reference cost and the field's gap are one mechanism,
assumed the gap was a fact about URP's field. It is a fact about the
instrument.
