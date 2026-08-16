# A palette of `d+1` maximises full-palette junction density in `d` dimensions

### — and that is a property of the measure, not of the field

> ## Retraction
>
> **This note originally presented `d+1` as a measurement about relaxed
> multiphase fields.** It is not. An outside review supplied the control the
> work never had: replace the entire apparatus — Allen–Cahn dynamics, capacity
> field, S-functional, relaxation protocol — with random points and
> nearest-neighbour colouring, and score the plain Voronoi diagram with this
> repository's own unmodified measure. **The argmax is still `d+1`, in every
> dimension.** It passes the texture guard. It has no dynamics of any kind.
>
> Worse for the original claim: **pure random labels** — no tessellation at all —
> also return `d+1`, at densities two orders of magnitude higher. That arm fails
> the texture guard, so it does not by itself refute a guarded claim; the Voronoi
> arm passes every check this repository applies, and does.
>
> The reason is structural, and it was visible in the definition the whole time.
> At `m = d+1` the measure demands all `P` sectors *and* at least `d+1` distinct
> ones in one `3^d` window, so every `P < d+1` is **identically zero by
> definition**, and every `P > d+1` must fit more colours into the same small
> window, which only gets harder. A statistic that is zero below `d+1` and
> decreasing above it peaks at `d+1` **before any field exists**.
>
> What survives is that the relaxed field tessellates space — which the texture
> guard already reported, and which was never in doubt. What does not survive is
> `d+1` as a finding about the field, as a result of this programme, or as a
> falsifier of anything in it. The sections below are kept as written, with the
> claim's actual standing marked, because the reasoning is worth reading and the
> record of having been wrong is part of what makes the rest checkable.
>
> Reproduce the control: `python experiments/n3_junction_null.py`

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

> **Standing: true, and empty of content about the field.** The statement holds.
> It also holds for a Voronoi diagram and for random noise, because — as the
> retraction above sets out — the measure is zero below `d+1` by construction and
> decreasing above it. Read what follows as *an analysis of a statistic*, which
> is what it turned out to be.

## Why `d+1` is the geometrically correct threshold

The condition `m` is not free. Codimension counting fixes it: where `m` sectors
meet, the locus they meet on has codimension `m − 1`. So

| `m` sectors meeting | locus in 1-D | in 2-D | in 3-D | in 4-D |
|---|---|---|---|---|
| 2 | **point** | line | plane | 3-volume |
| 3 | — | **point** | line | plane |
| 4 | — | — | **point** | line |
| 5 | — | — | — | **point** |

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

**`d = 4`** (lattice 24⁴, 1500 steps), `m = 5`, **two independent seed sets**:

| | `P=2` | `P=3` | `P=4` | `P=5` | `P=6` | `P=7` |
|---|---|---|---|---|---|---|
| run A | `0` | `0` | `0` | **0.000278** | `0` | `0` |
| run B | `0` | `0` | `0` | **0.000097** | `0` | `0` |

The *ranking* reproduces exactly — `P = 5` alone, every rival identically zero
in both. The *density* does not: it differs by a factor of `2.9` between seed
sets, because at this box size the whole reading rests on 30–90 qualifying cells.
Quote the ranking; do not quote the density to more than an order of magnitude.

In all four dimensions the argmax is `d + 1`. In `d = 1`, `2` and `4` every
rival is exactly zero. In `d = 3` the nearest rival (`P = 5`) is non-zero but
**10× smaller**, and `P = 3` — which wins in the plane — falls to identically
zero, because three sectors cannot meet at a four-fold vertex.

The `d = 4` arm is the weakest of the four and is reported as such below.

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

## What a `d = 4` tessellation is made of

Codimension counting predicts four tiers of structure in four dimensions rather
than three. All four are occupied (`P = 5`, fraction of cells at each valence):

| valence `k` | locus | run A | run B | falloff |
|---|---|---|---|---|
| 1 | domain interior | `0.579` | `0.579` | — |
| 2 | 3-D wall | `0.314` | `0.333` | `1.7–1.8×` |
| 3 | 2-D surface | `0.096` | `0.078` | `3.3–4.3×` |
| 4 | 1-D line | `0.0115` | `0.0104` | `7.4–8.3×` |
| 5 | **point vertex** | `0.00028` | `0.00010` | `41–107×` |

The falloff **accelerates** down the ladder. Point vertices are not merely the
rarest tier; they are rarer by a widening margin, which is the structural reason
this measurement gets harder with dimension rather than merely more expensive.

The two seed sets agree closely down to the line tier — within 20% — and
diverge by `2.8×` at the point tier. That is the honest shape of this result:
**the hierarchy is solid down to codimension 3 and seed-noisy at the vertex**,
which is exactly where the counting statistics run out.

### Are vertices scarcer at higher `d` because structure fails, or because points are small?

Mostly the second, and the distinction matters. Raw peak density falls `0.091 →
0.0069 → 0.0021 → 0.0001–0.0003` across `d = 1…4` — a 300–900× drop. But a point occupies
a vanishing fraction of a `d`-volume regardless of stability: for domains of
width `L`, vertex-containing cells should scale like `(3/L)^d` on geometry
alone. Dividing that out:

| `d` | domain width `L` | density × `L^d` | ratio to `3^d` |
|---|---|---|---|
| 1 | 22.0 | 1.99 | `0.66` |
| 2 | 19.5 | 2.64 | `0.29` |
| 3 | 16.8 | 10.10 | `0.37` |
| 4 | 15.7–16.2 | 6.68–16.89 | `0.08–0.21` |

The residual is a **3–8× decline, not 330×**, and it is not monotone — `d = 3`
sits above `d = 2`. So on this evidence **4-D point vertices form perfectly
well and are simply geometrically scarce.** Higher dimensions support this
structure; they just hold less of it per unit volume.

The `d = 4` entry is a range because the two seed sets disagree by `2.9×`, and
that range is wide enough to matter: it is the difference between "the residual
decline is negligible" and "the residual decline is real but small". Four
points, one of them a range spanning a factor of three, will not support a
stronger statement than *geometry dominates*.

## Scope, and what this is not

**Texture guard, and a correction to it.** A field that never coarsens into
domains would still produce neighbourhoods full of colours — lattice noise, not
a tessellation. The original guard was `domain_scale` (volume per unit wall
area) against a floor of `2.5`.

**That floor is dimension-blind, and it is the second such constant found in
this measure.** A cell counts as wall if any of its `3^d − 1` neighbours
differs, and that ring grows exponentially, so a larger share of any domain sits
within one step of its surface as `d` rises — the same raw scale means a smaller
domain in higher dimensions. Measured: 2-D at `scale = 5.14` and 4-D at
`scale = 2.38` hold domains of `19.5` and `15.7` lattice units. Comparable
structures, on opposite sides of the cut. `domain_diameter` inverts
`scale ≈ 1/(1 − ((L−2)/L)^d)` to report the width directly, and pure noise then
reads `1.0` in every dimension. Widths: `22.0` (1-D), `16.8–19.5` (2-D/3-D),
`15.7` (4-D) — all resolved.

**Disclosure on that fix.** The corrected guard is what makes the `d = 4` arm
readable at all: its raw scale of `2.38` is below the published floor, so the
old guard would have rejected it. I wrote the guard before the `d = 4` densities
existed — the commit precedes the run — and it is derived from the
interior-fraction geometry rather than fitted to anything. But an instrument
that licenses the result its author predicted is a pattern worth stating plainly
rather than leaving for a reader to notice.

**The `d = 4` arm is weak, specifically:** the box holds ~1.5 domains per axis
against 3.7 in the 2-D arm; 30–90 qualifying cells across all three trials,
which is why two seed sets differ by `2.9×`; and the `P = 6, 7` zeros cannot be
distinguished from under-sampling at this box size. Matching 2-D's resolution in four dimensions needs roughly `55⁴ ≈ 9M`
cells — about 30 hours — so this is a budget limit, not a measurement.

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

- ~~A palette other than `d + 1` scoring higher at `m = d + 1`, in any dimension,
  on a resolved field.~~ **Withdrawn.** This is not a falsifier: nothing about
  any field could make it come out differently, because the measure forbids it.
  Listing it as one was the original error in a single line.
- The `d = 3` result reversing under a different relaxation (the arm with a live
  rival is the vulnerable one).
- The `d = 4` arm reversing at a box large enough to hold a proper tiling. This
  is the most likely place for it to fail, because it is the least resolved.
- Any demonstration that `domain_diameter` is not distinguishing tessellation
  from texture, which would invalidate every arm at once.

`d = 5` is the frontier and is **not reachable by this approach**. Extrapolating
the tier falloff puts its point-vertex density near `1e-6`, below what any
affordable box can sample — so the failure there would be a measurement ceiling,
not evidence about the structure. Answering whether `d + 1` continues past four
dimensions needs a different instrument, not a bigger lattice.

## Reproducing it

```
python experiments/n3_junction_scale.py            # d = 1, 2, 3
python experiments/n3_junction_scale.py --with-4d  # adds d = 4 (~60 min)
```

The `d = 4` arm is opt-in because a 4-D box big enough to hold a tiling is
expensive; the default run stays affordable.

Prints its pre-registered predictions and every verdict, including the one that
failed. The measurement lives in `project_genesis/junction_scale.py`; the
`radius = 1`, `min_valence = None` case is pinned bit-identical to the original
published measure in `tests/test_junction_scale.py`, so widening the probe is
demonstrably widening *that* measure and not some other one.

---

*Both of this measure's constants turned out to be dimension-blind: the junction
threshold `m`, hardcoded to the 2-D vertex number, and the texture floor, tuned
to a 2-D domain. Finding the first was the original result — `d + 1` had been
asserted for some time and could not have been measured, because the instrument
used to look for it silently assumed the answer for `d = 2`. Finding the second
was what let the measurement reach four dimensions at all. A measure written in
one dimension will assume that dimension twice over before anyone notices.*
