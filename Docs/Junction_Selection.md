# A palette of `d+1` maximises full-palette junction density in `d` dimensions

### A self-contained measurement on relaxed multiphase fields

*This note stands alone. It assumes no framework, states one geometric claim,
reports what was measured, and says what would refute it. It is extracted from a
larger programme, but nothing here depends on that programme being right.*

## Where it stands (2026-10-31)

The sections below are in the order the work was done. In short:

- **The ranking holds.** At the point-junction test `m = d + 1`, the palette
  `P = d + 1` wins in `d = 1, 2, 3, 4`. `d = 4` is weak, resting on 30–90
  qualifying cells.
- **It is geometry, not dynamics.** A random Voronoi tessellation with no
  dynamics selects `d + 1` in `d = 2, 3, 4`. What the relaxation adds is
  genericity: degenerate junctions split, and relaxed fields are no less generic
  than random ones (2-D and 3-D; `d = 4` untested).
- **The published densities should not be quoted.** The published seeding left
  about 40% of each field at `−1`. With seeding that fills the volume, the
  ranking gets cleaner: the 3-D rival goes to zero. The 1-D and 4-D arms have
  not been re-run that way. See "The seeding problem".
- **`d = 5` is out of reach** for two reasons: point vertices too rare to
  sample, and a time step (`dt = 0.1`) that is unstable there.

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
`[0, 0.1]`, labelled by `argmax` over the sector index. *That seeding turned out
to be defective (see "The seeding problem"). The tables in this section are kept
as published; quote their rankings, not their densities.* Three trials per palette.
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

## Geometry or dynamics? A null with no dynamics at all

The rivals *below* `d + 1` are zero by counting. That leaves two questions:
what keeps the rivals *above* `d + 1` down, and do the relaxation dynamics do
any of the work? `experiments/n3_junction_null.py` checks three things.

**N1: a random Voronoi tessellation already selects `d + 1`.** Random seeds,
random colours, no energy, no conservation, same probe, `m = d + 1`:

| `d` | box | `P=d−1` | `P=d` | `P=d+1` | `P=d+2` | `P=d+3` | margin |
|---|---|---|---|---|---|---|---|
| 2 | 160², 70 seeds | `0` | `0` | **0.00562** | `0.00016` | `0` | `36×` |
| 3 | 56³, 40 seeds | `0` | `0` | **0.00246** | `0.00013` | `0` | `18.5×` |
| 4 | 26⁴, 24 seeds | `0` | `0` | **0.00437** | `0.00097` | `0.00004` | `4.5×` |

So "geometric, not energetic" holds in the strongest form available: the
selection appears with **no dynamics at all**. The other side of that result is
that the ranking in the relaxed fields is not *evidence about* the dynamics. It
is what any generic tessellation does. The non-zero `P = d + 2` readings here
are lattice artefacts: a `3^d` window can straddle two nearby generic vertices.
They are also a caution about the relaxed `d = 4` table. Its rivals reading
exactly zero is a sampling floor at 30–90 qualifying cells, not a property of
the field.

**N2: point junctions are not rare for large palettes.** Drop the full-palette
condition and count every neighbourhood with `≥ d + 1` sectors. The density is
non-zero for every `P ≥ d + 1` and **rises** with `P` (2-D: `0.0056 → 0.0090 →
0.0118 → 0.0140` for `P = 3…6`). What `d + 1` selects is not *whether point
junctions exist*. It is whether a point junction carries **the whole palette**.
`P = d + 1` is the only palette whose generic vertices are always
palette-complete.

**N3: the dynamics' real contribution is removing degenerate vertices.** Seed
a 2-D field as a 4-colour checkerboard of `18×18` blocks, so every vertex is
four-fold and `P = 4` scores at a vertex. Then relax it with the published
dynamics:

| step | 0–600 | 700 | 800 | 1000 | 1200 | 1500 | 2000 |
|---|---|---|---|---|---|---|---|
| four-fold density | `0.01235` | `0.01042` | `0.00424` | `0.00154` | `0.00077` | `0` | `0` |
| triple-junction cells | 64 | 82 | 122 | 124 | 142 | 138 | 138 |

Phase volumes stay conserved to within 1%, and the domain width stays at
`16–18`. The four-fold vertices hold as a metastable state for about 600 steps,
then split into triple junctions, and the triple-junction count roughly
doubles. That is Plateau's law operating, and it is the one part of the result
that a random null cannot supply. Voronoi vertices are generic *by chance*.
Relaxed vertices are generic *because degenerate ones are unstable*. The
selection therefore holds for any tessellation that is generic, and the
dynamics are one way, not the only way, of getting there.

## Are the relaxed fields actually generic?

The null reduces the claim to one assumption: relaxed fields are *generic*. No
point carries more than `d + 1` sectors, and no edge carries more than its
codimension allows. `experiments/n3_junction_genericity.py` tests that
assumption directly, using two instruments, and reads both against a Voronoi
baseline at matched domain width. Voronoi is generic with probability one, so
whatever it scores is lattice artefact and nothing else.

- **`excess`** is, among cells with `≥ d + 1` sectors in view, the share with
  `≥ d + 2`. It catches over-valent *points*.
- **`blob`** is the mean connected-component size of those cells. It catches
  over-valent *edges*, which join vertex blobs into extended runs.

The second instrument was added mid-run, after the pre-registered positive
control **failed** in 3-D. An 8-colour cubic checkerboard, which is degenerate
by construction, scored an `excess` of only `0.10`. Its four-fold *edges* put
valence-4 cells along whole lines, and `excess` reads those as ordinary
vertices. That blindness matters, because an over-valent edge is exactly what
would let a non-generic field inflate the `P = d + 1` count. `blob` sees it at
once: the checkerboard reads `3430–5120` cells against Voronoi's `14–29`.

| `d` | `P` | relaxed `excess` | Voronoi `excess` | relaxed `blob` | Voronoi `blob` |
|---|---|---|---|---|---|
| 2 | 3 | `0` | `0` | 4.6 | 4.6 |
| 2 | 4 | `0.0054` | `0.0137` | 5.1 | 4.8 |
| 2 | 5 | `0.0309` | `0.0296` | 5.6 | 5.2 |
| 2 | 6 | `0.0148` | `0.0260` | 4.6 | 5.0 |
| 3 | 4 | `0` | `0` | 15.0 | 14.1 |
| 3 | 5 | `0` | `0.0193` | 12.6 | 17.0 |
| 3 | 6 | `0.0307` | `0.0369` | 14.0 | 22.2 |
| 4 | 5 | — (0 junction cells) | `0` | — | 100 |
| 4 | 6 | `0` (n = 207) | `0.0924`† | 25.9 | 656† |

† **Not a matched baseline.** A `24⁴` box cannot hold Voronoi cells at the
relaxed width (`16.7`): the seed count bottoms out at the palette size, and the
baseline runs at width `9.9`, where neighbouring vertices crowd into each other's
windows. The `d = 4` arm therefore shows only that the one relaxed `P = 6` field
has **no over-valent points among 207 junction cells**. The single `P = 5` field
had no junction cells at all. The 4-D checkerboard control reads `excess = 1`
and `blob = 36864`, so the instruments do work there. What's missing is the
statistics and a box large enough to match. Treat `d = 4` as not yet tested.

On both instruments, the relaxed fields are **no less generic than random
Voronoi** at every palette tested. They are often more generic, which fits
degenerate contacts being actively unstable rather than merely improbable.

**The 3-D degenerate start splits.** The 8-colour checkerboard (`32³`, block 8),
relaxed with the published dynamics:

| step | 0–300 | 800 | 1500 | 2500 |
|---|---|---|---|---|
| valence `≥ 5` (impossible at a generic 3-D point) | `0.0156` | `0.00095` | `0.00024` | `0` |
| `blob` | 5120 | 18.4 | 13.7 | 12.8 |
| domain width | 8.0 | 10.0 | 11.9 | 12.2 |

The eight-fold vertices and four-fold edges both dissolve, and the field ends
with blobs the size of Voronoi's. This is the 2-D checkerboard result, one
dimension up, and it covers the defect that 2-D could not show: degenerate
*edges* as well as degenerate points.

**Limits.** These boxes are small (`72²`, `28³`), with 2–3 trials. The Voronoi
match for the relaxed checkerboard is loose (width 9.3 against 12.2). The
narrower width should, if anything, *raise* the baseline's artefact rate. And
"no less generic than Voronoi" is a lattice-level statement: it bounds
degeneracy to below what the `3^d` probe can resolve, not to zero.

## The seeding problem

The published seeding, uniform noise on `[0, 0.1]`, conserves each sector's mean
at `0.05`. For any `P ≤ 20` that is far too little volume to tile space at `+1`.
The relaxed field makes up the balance with sectors sitting at **`−1`**:

| field | `+1` cover | `−1` only | median argmax margin in `−1` regions |
|---|---|---|---|
| `P = 3`, 2-D | 0.563 | **0.415** | 0.027 |
| `P = 4`, 2-D | 0.595 | **0.385** | 0.007 |
| `P = 4`, 3-D | 0.540 | **0.397** | 0.017 |

In those regions every other sector is near zero, and `argmax` picks a label
from residuals of order `0.01`. So about 40% of every published field carries a
label that is not a phase. In the 3-D `P = 4` field, two of the four sectors
barely reach `+1` at all (`0.5%` and `3.3%` of cells).

With **volume-filling seeding** (noise on `[0, 2/P]`, so each mean is `1/P`),
the `−1` regions vanish (`+1` cover `0.955–1.000`, `−1` only `0`). The selection
then gets cleaner:

| | `P=2` | `P=3` | `P=4` | `P=5` | `P=6` |
|---|---|---|---|---|---|
| 2-D, `m = 3` (72², 3 trials) | `0` | **0.01530** | `0` | `0` | `0` |
| 3-D, `m = 4` (28³, 2 trials) | `0` | `0` | **0.00271** | `0` | `0` |

The 3-D `P = 5` rival, live at `10×` behind in the published table, is now
identically zero. So the published argmax survived a labelling defect rather
than depending on it. The published *densities* are not quantities of a clean
tessellation, though, and should not be quoted as such. The tables above this
section are left as published, for the record.

## What would refute it

- A palette other than `d + 1` scoring higher at `m = d + 1`, in any dimension,
  on a resolved field.
- The `d = 3` result reversing under a different relaxation (the arm with a live
  rival is the vulnerable one).
- The `d = 4` arm reversing at a box large enough to hold a proper tiling. This
  is the most likely place for it to fail, because it is the least resolved.
- A relaxed field scoring clearly above a matched Voronoi baseline on `excess`
  or `blob` (`experiments/n3_junction_genericity.py`). That would mean the
  dynamics stabilise degenerate contacts and the zeros above `d + 1` are
  partly luck of sampling.
- Any demonstration that `domain_diameter` is not distinguishing tessellation
  from texture, which would invalidate every arm at once.

`d = 5` is the frontier and is **not reachable by this approach**. Extrapolating
the tier falloff puts its point-vertex density near `1e-6`, below what any
affordable box can sample — so the failure there would be a measurement ceiling,
not evidence about the structure. There is a second barrier as well. With
`dt = 0.1`, a grid-scale wiggle near a pure phase is multiplied by
`1 − dt·(4d + 2)` per step. That factor is `−0.8` in 4-D and `−1.2` in 5-D, so
the relaxation itself is unstable there (`Docs/Constants_Audit.md`). Answering whether `d + 1` continues past four
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
