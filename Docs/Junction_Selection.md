# A palette of `d+1` maximises full-palette junction density in `d` dimensions

### A self-contained measurement on relaxed multiphase fields

*This note stands alone. It assumes no framework, states one geometric claim,
reports what was measured, and says what would refute it. It is extracted from a
larger programme, but nothing here depends on that programme being right.*

---

## The claim

Take a multiphase field with `P` distinguishable sectors, relax it under
volume-conserving dynamics until it partitions space into domains, and ask a
local question of every cell:

> Does this cell's immediate neighbourhood contain **all `P` sectors at once**,
> and does it contain at least `m` distinct sectors?

Call the fraction of cells answering yes the **full-palette junction density**.
The claim is that in `d` spatial dimensions, with `m` set to the number of
sectors that meet at a genuine point junction, this density is maximised at

```
    P = d + 1
```

and that every other palette is not merely worse but **identically zero**.

## Why `d+1` is the geometrically correct threshold

The condition `m` is not free. Codimension counting fixes it: where `m` sectors
meet, the locus they meet on has codimension `m − 1`. So

| `m` sectors meeting | locus in 1-D | in 2-D | in 3-D |
|---|---|---|---|
| 2 | **point** | line | plane |
| 3 | — | **point** | line |
| 4 | — | — | **point** |

A *point* junction in `d` dimensions therefore requires exactly `d + 1` sectors.
Fewer, and the sectors meet along an extended locus — a wall or an edge, not a
vertex. This is the same counting that gives Plateau's laws their trivalent
vertices in 2-D soap films.

The consequence that makes the claim sharp: **a palette of `P` sectors cannot
form a junction of order greater than `P`.** So at `m = d + 1`, every palette
below `d + 1` is excluded by counting alone, and the measurement is only asking
whether larger palettes do better. They do not.

## What was measured

Fields of `P` sectors relaxed under a volume-conserving multiphase Allen–Cahn
flow (`γ = 1.5`, `dt = 0.1`, unit diffusion), seeded from uniform noise on
`[0, 0.1]`, labelled by `argmax` over the sector index. Three trials per palette.
Neighbourhood is the immediate `3^d` ring — the smallest window in which a
junction can appear at all.

**`d = 1`** (lattice 4000, 1500 steps), `m = 2`:

| | `P=2` | `P=3` | `P=4` | `P=5` | `P=6` |
|---|---|---|---|---|---|
| density | **0.09067** | `0.00000` | `0.00000` | `0.00000` | `0.00000` |

**`d = 2`** (lattice 72², 600 steps), `m = 3`:

| | `P=2` | `P=3` | `P=4` | `P=5` | `P=6` |
|---|---|---|---|---|---|
| density | `0.00000` | **0.00694** | `0.00000` | `0.00000` | `0.00000` |

**`d = 3`** (lattice 28³, 1500 steps), `m = 4`:

| | `P=2` | `P=3` | `P=4` | `P=5` | `P=6` |
|---|---|---|---|---|---|
| density | `0.00000` | `0.00000` | **0.00213** | `0.00021` | `0.00000` |

In all three dimensions the argmax is `d + 1`. In `d = 1` and `d = 2` every
rival is exactly zero. In `d = 3` the nearest rival (`P = 5`) is non-zero but
**10× smaller**, and `P = 3` — which wins in the plane — falls to identically
zero, because three sectors cannot meet at a four-fold vertex.

### The control that matters

The same 3-D field scored with the *2-D* condition `m = 3` selects `P = 3`, not
`P = 4`:

| condition | `P=2` | `P=3` | `P=4` | `P=5` | `P=6` | argmax |
|---|---|---|---|---|---|---|
| `m = 3` (2-D vertex) | `0.00000` | `0.02902` | `0.00213` | `0.00021` | `0.00000` | `3` |
| `m = 4` (3-D vertex) | `0.00000` | `0.00000` | `0.00213` | `0.00021` | `0.00000` | `4` |

This is the whole result in one table. A dimension-blind threshold of `3` asks,
in 3-D, whether an **edge** carries the full palette — which a 3-palette does
trivially, along the line where its three sectors meet. Fixing the threshold to
the dimension is what makes `d + 1` visible; it was invisible before, not
because the geometry was absent but because the instrument could not express it.

## Scope, and what this is not

**Texture guard.** A field that never coarsens into domains would still produce
neighbourhoods full of colours — lattice noise, not a tessellation. Every arm
above is checked with `domain_scale` (volume per unit wall area), which reads
`4.4–9.8` in 2-D and `7.8–11.0` in 1-D against a floor of `2.5`. These are
resolved domains.

**This is a categorical result, not a robustness result, and the distinction
matters.** Where the rivals are *identically zero*, no perturbation of the
convention could have changed the ranking — the claim is true by counting, and
the measurement confirms the counting rather than testing it. That is a stronger
statement than robustness, but it is a different one, and reporting it as though
a sweep had threatened it would be wrong. The one place a sweep does bite is the
`d = 3` arm, where `P = 5` is live at `10×` behind.

**The neighbourhood is the one free choice**, and widening it degrades the
result as expected: in 2-D the margin over the nearest rival falls `∞ → 45.5 →
16.6` as the radius goes `1 → 2 → 3`. A radius-2 window is `5×5` and can hold
four colours even where every vertex is strictly three-fold, so it measures a
*region* rather than a vertex. The honest form of the claim is therefore
*"`d + 1` is selected relative to a vertex-scale probe"* — which is what a
statement about vertices should be relative to. The **ranking** survives every
radius tested while the **margin** does not.

**Three trials per palette**, one relaxation protocol, one seeding scheme. This
is not a study of universality across dynamics.

## What would refute it

- A palette other than `d + 1` scoring higher at `m = d + 1`, in any dimension,
  on a resolved field.
- The `d = 3` result reversing under a different relaxation (the arm with a live
  rival is the vulnerable one).
- `d = 4`, which is untested here and is the obvious next check. The claim
  predicts `P = 5` at `m = 5`.
- Any demonstration that `domain_scale` is not distinguishing tessellation from
  texture, which would invalidate every arm at once.

## Reproducing it

```
python experiments/n3_junction_scale.py
```

Prints its pre-registered predictions and every verdict, including the one that
failed. The measurement lives in `project_genesis/junction_scale.py`; the
`radius = 1`, `min_valence = None` case is pinned bit-identical to the original
published measure in `tests/test_junction_scale.py`, so widening the probe is
demonstrably widening *that* measure and not some other one.

---

*The `m = 3` constant was hardcoded and dimension-blind. Finding that was the
whole result: `d + 1` had been asserted for some time and could not have been
measured, because the instrument used to look for it silently assumed the
answer for `d = 2`.*
