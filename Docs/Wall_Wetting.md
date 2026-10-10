# Does a wall make room for a new distinction?

### A wetting test on the multiphase field

*Self-contained. Code: `project_genesis/wetting.py`, `experiments/n3_wetting.py`,
`tests/test_wetting.py`.*

---

## The question

Start from the simplest partition: two sectors, A and B, relaxed into domains.
Between them sits a wall, the one place where neither is fully on. Add a third
sector C everywhere at amplitude `≤ 0.01`. Let it grow freely (not
volume-conserved), keep A and B conserved, and ask whether C switches on, and
where.

The published model cannot ask this, for two reasons. Every pair of sectors
shares one coupling `γ = 1.5`, so every wall costs the same, and a third sector
can never cheapen an A|B wall. Every sector is also volume-conserved, so a
sector seeded near zero can never grow. `step_multiphase_pairwise` makes the
coupling a matrix `γ_ab` and conservation a per-sector choice. With uniform
couplings and everything conserved, it reproduces the published step to `1e-12`
(pinned in the tests).

Here `γ_AB = 1.5`, as published, and `γ_AC = γ_BC = g` is the dial.

## Why the wall should matter

A sector at zero grows at the linear rate `λ = 1 − 2 Σ_b γ_cb f_b²`. In bulk A,
that is `1 − 2g`, so C is held at zero exactly when `g > 1/2`. In an A|B wall,
both `f_A` and `f_B` are below one, so the same suppression is weaker. A
distinction the bulk forbids is least forbidden in the grayscale between two
others.

## Result

`96²`, three seeds, 600 steps to relax A|B, then 2000 steps with C. *Onset* is
mean `|f_C|` on the original A|B walls divided by mean `|f_C|` off them, at step
150, before anything saturates. *Contact* is A–B wall remaining, relative to a
no-C control (`g = 1.5`).

| `g` | C max | C cover | contact | 3-sector junctions | onset | regime |
|---|---|---|---|---|---|---|
| 0.30 | 0.82 | 1.00 | 0 | 0 | 1.2 | C everywhere |
| 0.45 | 0.77 | 0.25 | 0 | 0 | 3.6 | complete wetting |
| 0.50 | 0.78 | 0.17 | 0 | 0 | 4.2 | complete wetting |
| 0.52 | 0.76 | 0.15 | 0.009 | 0.0007 | 4.3 | partial wetting |
| 0.55 | 0.73 | 0.12 | 0.10 | 0.0077 | 4.6 | partial wetting |
| 0.58 | 0.67 | 0.08 | 0.33 | 0.0251 | 4.8 | partial wetting |
| 0.60 | 0.65 | 0.05 | 0.62 | **0.0420** | 5.0 | partial wetting |
| 0.62 | 0.43 | 0.00 | — | 0.0011 | 5.1 | marginal |
| 0.65–1.5 | `≤ 0.002` | 0 | 1 | 0 | 5.3–9.4 | dormant |

(0.50–0.62 except 0.55 were added after a `--quick` run located the threshold.
They are post hoc, and the script labels them so.)

**Four regimes, in order of increasing `g`:**

1. **C everywhere (`g ≈ 0.3`).** The bulk cannot hold C at zero, so C mixes in
   everywhere.
2. **Complete wetting (`g ≲ 0.5`).** C forms a continuous layer along every A|B
   wall. A and B never touch again, so no three-sector point junction can
   exist. The density is identically zero.
3. **Partial wetting (`0.52 ≤ g ≤ 0.60`).** C grows into the walls but does not
   seal them. Where a C layer ends, A, B and C meet at a point, giving
   three-sector junctions. This is the only window in which they appear.
4. **Dormant (`g ≥ 0.65`).** C never takes off. It includes the published
   coupling `g = 1.5`.

**Growth starts on the walls at every coupling.** Onset enrichment is `3.6–9.4×`
for every `g ≥ 0.45`. Even where C never takes off, what is left of the seed
survives longest in the walls. Only at `g = 0.3`, where the bulk is itself
unstable, does the enrichment approach 1.

**So, the direct answer.** Yes, the wall makes room. In a narrow window it makes
room for exactly what the plane wants, three sectors meeting at a point. The
field reaches its `d + 1` palette by growing into it rather than being handed
it. At `g = 0.60` the grown field's junction density (`0.042`) is about `2.7×`
that of a 3-sector field relaxed from noise (`0.0153`). The geometry differs,
though, because the junctions sit at the ends of C layers along the old walls,
so that ratio is not a like-for-like comparison.

## Pre-registered verdicts

| | prediction | verdict |
|---|---|---|
| W1 | for `g < 0.5`, C covers more than `3×` the wall fraction | **fails** at `g = 0.45` (`0.25` against `0.34`), holds at `0.3` |
| W2 | a wall-only growth window: final enrichment `> 3` and contact `< 1/2` | **fails as specified** |
| W3 | dormant at `g = 1.5` | holds |
| W4 | three-sector junctions somewhere in the growth window | holds |

**W1** was too strong. At `g = 0.45` the bulk rate is only `+0.1`, and the
conserved A and B domains crowd C into wall layers well before it can invade
the bulk. Bulk invasion needs `g` clearly below `1/2`, not just below it.

**W2** fails for two reasons, both in the specification, not in the field. The
first is the window. I set it as `(0.5, g_lin)` using the local linear
threshold, `g_lin = 1.00`. The actual threshold is near `0.62`, because a layer
one wall thick pays a gradient cost that the local estimate ignores. Most of the
pre-registered window was therefore dormant. The second is the measure. I
scored enrichment on the *final* state against the *initial* wall mask, but a
grown C layer is thicker than that mask, which caps final enrichment near
`1.5`. Read at onset, enrichment is `4–5×` across the window. That reading is
post hoc, and the table above is the evidence for it, not the verdict.

## What this does and does not show

- **C was in the equations from the start.** This shows that a wall makes
  *room* for a distinction already latent in the system. It does not show a
  distinction appearing from nothing.
- **`g` is a dial I set.** The window `0.52–0.60` is narrow, and nothing here
  says why a system would sit inside it. Asking that would mean letting `g`
  itself respond to the field.
- **One dimension, one box, three seeds.** `2-D`, `96²`. The obvious extension
  is `3-D` with a fourth sector added to a 3-sector field: does partial wetting
  there give four-sector point junctions, the `d + 1` palette one dimension up?

## Seeding note

A and B are seeded with volume-filling noise (mean `1/2` each). The published
seeding (noise on `[0, 0.1]`) conserves a mean of only `0.05` per sector. That is
too little to tile space at `+1`, and the field makes up the balance with
sectors sitting at `−1`, about 40% of the box. Those regions have `f ≈ 0` for
every other sector, so a dormant C would see them as walls. See
`Junction_Selection.md`, "The seeding problem", for what this does to the
published tables. In short, the selection survives and gets cleaner.
