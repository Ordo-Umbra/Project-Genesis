# Constants audit: the multiphase field stack

*Started 2026-10-31. Built from reading the code and docs; no experiment was
re-run. One derivation (the time step) was checked numerically against the real
update. Where a status says **at risk**, the reason was worked out on paper and
has not been measured on a relaxed field.*

The question for every constant: **where was it chosen, where is it used now,
and has anyone checked it outside the place it was chosen?** This is the open
item `CLAUDE.md` names, after the same failure type turned up three times.

---

## The answer first

24 constants in the field stack. Almost all were chosen in **2-D with three
sectors**. Outside that setting, every one that has been tested has either held
or broken; most have not been tested at all.

| status | meaning | count |
|---|---|---|
| **live** | known to be wrong, still in use where it is wrong | 3 |
| **broken** | shown wrong outside its home regime (on record) | 3 |
| **at risk** | a derived reason to expect trouble; not measured | 2 |
| **checked** | tested outside its home regime, and held | 4 |
| **derived** | fixed by geometry or analysis | 2 |
| **unchecked** | never tested outside its home regime | 8 |
| **known limit** | a recorded weakness, not a hidden one | 2 |

## Five findings, most consequential first

**1. The 2-D junction threshold is still live in 3-D, in three current documents.**
`The_Principle.md` §6, which calls itself "the falsifiable core", says that in
3-D "`P = 3` still wins". `Narrowing_the_N3_Question.md` says three wins in 3-D
"~10×". The `n3_selection_sweep.py` docstring and `Usage.md` say `P = d + 1` is
"ruled out". All of these readings use `multiphase.full_palette_junction_density`,
which hardcodes `distinct >= 3`. In 3-D, three sectors meet along a *line*, so
that test asks whether an **edge** carries the whole palette. A three-sector
field passes it trivially. The repo already measured this: the control table in
`Junction_Selection.md` shows the same 3-D field selecting `P = 3` at `m = 3`
and `P = 4` at `m = 4`. So the 3-D half of §6 reads edges, not vertices, and
contradicts §3 of the same document. The 2-D readings are unaffected.

**2. The seeding defect reaches six experiments, not one.** All six pair
`rand × 0.1` seeding with volume-conserving dynamics:

| experiment | line | what rests on it |
|---|---|---|
| `n3_junction_scale` | 94 | the published `d + 1` tables (re-run clean in 2-D/3-D) |
| `n3_capacity_separation` | 86 | capacity invariance of φ (supports the README row) |
| `topological_selection` | 57 | `S` maximised at three sectors |
| `n3_thermal_selection` | 129 | selection under a thermal gauge field |
| `gauge_coherence` | 94 | sector field used as a backdrop for gauge demos |
| `yang_mills_flow` | 81 | wall modes read with `interface_mask` |

The README's capacity row rests mainly on `n3_capacity_gating`, which uses
`selection.py`. That code evolves *non-conserved* dynamics from `uniform(0, 1)`,
where small seeds can grow, so it is **not** affected. The two κ-gated
experiments (`multiphase_kappa`, `standing_integration`) also use non-conserved
dynamics and are not affected.

**3. `interface_mask` will read the `−1` regions as wall.** It marks a cell as
wall when the gap between its top two components is below `0.15`. In the `−1`
regions that published seeding leaves (about 40% of each field), the median gap
between the top two components was measured at `0.007–0.027`
(`Junction_Selection.md`, "The seeding problem"). So under that seeding, most of
those regions should count as wall. `yang_mills_flow` and `n3_thermal_selection`
use both the mask and the seeding. This is derived from measured numbers. The
mask itself has not been run on such a field.

**4. The time step `dt = 0.1` nears its stability limit as dimension rises.**
For forward Euler on this flow, a grid-scale wiggle near a pure phase is
multiplied by `1 − dt·(4d + 2)` each step: `4d` from the Laplacian and `2` from
the stiffness of the wells. Checked against `step_multiphase_conserved` itself,
to three decimals, for `d = 1` to `5`:

| `d` | factor per step | meaning |
|---|---|---|
| 1 | `+0.4` | damped |
| 2 | `0.0` | killed in one step |
| 3 | `−0.4` | flips sign, damped |
| 4 | `−0.8` | flips sign, loses only 20% per step |
| 5 | `−1.2` | **grows: unstable** |

The `d = 4` arm reads `3⁴` neighbourhoods, which is the scale this ringing
lives at. The junction runs inject no noise, so after 1,500 steps such modes
have decayed (`0.8¹⁵⁰⁰` is negligible), and the relaxed end states are probably
unaffected. Any 4-D run that injects noise every step would be affected. This
also gives a second, independent reason `d = 5` is out of reach: at this time
step the field cannot be relaxed there at all.

**5. The failure-type table in `CLAUDE.md` was short by one entry already on
record.** The κ recovery default `r = 0.1` sits below a knee. It caused the
memory-recall collapse (2-D) and starved the form-persistence field into
texture (3-D). Both are in `Experiment_Log.md`. It is still the code default.

### The pattern behind them

Every dimension or regime fix so far was added **as an opt-in, leaving the old
default in place**:

| defect | fix | old default still in |
|---|---|---|
| `m = 3` | `min_valence=` in `junction_scale` | `multiphase.full_palette_junction_density`, `count_triple_junctions`, `selection.py` |
| floor `2.5` | `domain_diameter` | `resolved_palette_junction_density(min_scale=2.5)`, two experiments |
| seeding | `× 2/P` in new experiments | six experiments |
| κ recovery | `r = 0.8` passed explicitly | `step_multiphase_conserved`, `step_multiphase_kappa` |

This follows from working principle 3 ("pin before you extend"), and the
principle is right: published results must stay reproducible. Its side effect
is that a fixed defect stays live for every caller that doesn't opt in, and
older documents are never re-read against the fix. Nothing in the workflow
sends a fix back to the claims made before it. That is the systemic part.

---

## The full table

Line numbers are for the commit this audit was written against.

### Relaxation

| # | constant | home → also used in | status |
|---|---|---|---|
| R1 | coupling `γ = 1.5` (`multiphase.py:44`, `:94`; `wetting.py`) | 2-D three-colour toy → every `d`, every `P` | **unchecked** |
| R2 | diffusion `D = 1` | 2-D → every `d` | **unchecked** |
| R3 | time step `dt = 0.1` | 2-D → 1-D to 4-D | **at risk** (`d ≥ 4`) |
| R4 | wells at `±1` for each component | the potential's form | **derived** |
| R5 | conservation by subtracting the mean drift | 2-D → every `d` | **unchecked** |
| R6 | κ recovery `r = 0.1` (`multiphase.py:97`, `:156`) | unrecorded → 2-D, 3-D | **broken** |
| R7 | κ consumption `5.0`, κ diffusion `0.5`, baseline `1.0` | unrecorded | **unchecked** |
| R8 | ΔC weight `β = 0.09` | imported from earlier claims | **checked** (as a ratio) |

- **R1.** A pure phase is stable against a second component exactly when
  `γ > 1/2`. The docstring says `γ > 1`, which is sufficient but not the
  threshold. `Wall_Wetting.md` confirms `1/2`. The junction selection has never
  been run at another `γ`; only the wetting experiment varies a coupling, and
  that is a different one (`γ_AC`).
- **R2.** With `D = 1` and well stiffness `2`, the decay length of a wall is
  `√(D/2) ≈ 0.7` cells, so walls are only a few cells wide in every dimension.
  Whether coarsening is slowed by the lattice itself (walls pinned to the grid)
  is untested.
- **R4.** `¼η⁴ − ½η²` has the same energy at `η = −1` as at `+1`, so a sector
  sitting at `−1` costs nothing extra in bulk. That is why bad seeding could
  hide: conservation chose `−1` to make up the missing volume, and nothing
  penalised it.
- **R5.** Docstrings describe the conserved field as settling into a "stable
  tiling". The measured 3-D domain scale still rose between 600 and 1,500 steps
  (from `2.34–2.45` to `2.72–3.17`, `n3_junction_scale.py:129`), so at least
  there the field was still coarsening.
  Densities are snapshots at a chosen time (see S5).
- **R6.** On record in `Experiment_Log.md` (recall rescue; form-persistence
  backfill). Two more experiments run their κ arms at this default:
  `n3_capacity_separation` and `n3_capacity_gating`, the latter through
  `selection.make_stepper`. `n3_capacity_gating` reported that "scarcity
  destroys the selection", and that only some of its scarce arms passed the
  texture guard. That is what a field starved below the knee would look like.
  It hasn't been traced. The README's conclusion (the selection is geometric)
  doesn't depend on it, because the Voronoi null now carries that claim.
- **R8.** The value comes from claims that `The_Principle.md` §7 declines. In
  the selection runs `S = β·load + κ·w·neutrality`, so the ranking depends only
  on `w/β`, and `w` is swept. The absolute value carries nothing.

### Seeding and run length

| # | constant | home → also used in | status |
|---|---|---|---|
| S1 | seed `rand × 0.1` | never checked → six experiments | **live** |
| S2 | volume-filling seed `rand × 2/P` | 2-D, 3-D | **checked** (2-D/3-D only) |
| S3 | degenerate start `0.9·indicator + rand × 0.01` | 2-D, 3-D checkerboards | **unchecked** |
| S4 | dormant sector seed `0.01` (`n3_wetting.py:92`) | 2-D wetting | **unchecked** |
| S5 | steps: 600 (2-D), 1,500 (1-D, 3-D, 4-D) | — | **unchecked** |
| S6 | boxes `72²`, `28³`, `24⁴`; 2–3 trials | — | **known limit** |

- **S2.** `+1` cover is `0.955–1.000` with no `−1` regions, in 2-D and 3-D. The
  1-D and 4-D arms haven't been re-run with it (session log, open item 2).
- **S3.** Sector means sum to `0.9 + 0.005·P`: `0.92` at `P = 4`, `0.94` at
  `P = 8`. That is slightly short of the `1.0` needed to tile at `+1`. Published
  seeding falls short by 70–90% (`P = 2–6`), so this is small, but whether it
  leaves any `−1` cells has not been looked at.
- **S5.** The 3-D count of 1,500 was chosen so the fields would clear the raw
  `2.5` floor (G1), which is itself a broken constant. A longer run only
  coarsens further, so this probably did no harm. Whether the ranking holds at
  other step counts has not been tested.

### Measurement

| # | constant | home → also used in | status |
|---|---|---|---|
| M1 | junction valence `m = 3` (`multiphase.py:371`, `:394`; `junction_scale.py:111`, `:128`) | 2-D → 3-D | **live** |
| M2 | neighbourhood radius 1 (`3^d` ring) | 2-D → 3-D, 4-D | **checked** (2-D, 3-D) |
| M3 | von Neumann window (`2d + 1` cells) in `dimensional_forms.local_dimension` | 2-D → 3-D | **unchecked** |
| M4 | labelling by `argmax` | everywhere | **broken** (under S1) |
| M5 | wall margin `0.15` (`multiphase.py:340`) | 2-D → five experiments | **at risk** |

- **M1.** Correct only in `junction_scale.full_palette_density(min_valence=d+1)`.
  It is still `3` in `full_palette_junction_density`, `count_triple_junctions`,
  `topological_s_functional`, `resolved_palette_junction_density`,
  `valence_profile`'s `junction_fraction`, and so in `selection.py`. Those are
  reached in 3-D by `n3_selection_sweep --ndim 3`,
  `topological_selection --dim 3` and `n3_capacity_gating --ndim 3`. Finding 1
  covers the docs.
- **M2.** Radii 1–3 in 2-D and 3-D: the ranking survives, the margin does not
  (`Junction_Selection.md`). Not swept in 4-D.
- **M3.** The census counts sectors in the `2d + 1` von Neumann window, while
  the junction measures use the `3^d` Moore ring. The census formula itself is
  dimension-aware (`d + 1 − k`). Whether its counts depend on the window choice
  is untested.
- **M4.** Fine when every cell has one component near `+1` (S2). Under S1 about
  40% of cells are labelled from residuals of order `0.01`. On record.
- **M5.** Finding 3. Callers: `yang_mills_flow`, `n3_thermal_selection`,
  `n3_annealed_matter`, `n3_kappa_criticality`, `analyze_multiphase`. The
  seeding of the last three was not checked here.

### Guards

| # | constant | home → also used in | status |
|---|---|---|---|
| G1 | raw `domain_scale ≥ 2.5` (`multiphase.py:490`) | 2-D → 3-D, 4-D | **live** |
| G2 | `domain_diameter` | derived for compact domains, any `d` | **checked** |
| G3 | diameter `≥ 3` cells (form persistence) | from the `3^d` probe | **derived** |
| G4 | majority filter, one pass | 2-D | **known limit** |
| G5 | `n_phases` as a texture guard | 3-D form persistence | **broken**, replaced |

- **G1.** Still the default in `resolved_palette_junction_density`. Still used
  raw in `n3_junction_scale.py:209–211` (2-D/3-D arms) and
  `n3_selection_sweep.py:116`. In 3-D the raw scale falls at a fixed domain
  width, so this floor is *stricter* there than intended. It errs toward
  rejecting real tilings, not passing texture. That makes it low-risk where it
  still runs, but it is still the wrong guard.
- **G2.** Pure noise reads `1.0` in 2-, 3- and 4-D. It over-reads slabs by
  about a factor of `d`, as documented in its docstring.
- **G4.** On noise one pass takes the density from `0.997` to `0.641`. It helps,
  and it is not a guard on its own. On record.

### Verdict thresholds

These are pre-registered choices, not constants inherited from another regime,
so they are listed rather than scored. Each is stated in its own script.

- **Genericity:** relaxed `≤ 1.5×` Voronoi; at least 20 junction cells; positive
  control `excess > 0.5`; Voronoi matched at `0.8 / 1.0 / 1.2×` the relaxed
  width. In 4-D no matched baseline fits the box (on record).
- **Null:** seed counts set for domain width 15–20; N3 asks for a drop below
  10%.
- **Wetting:** C counts as "on" above `0.5`; onset is read at step 150;
  enrichment `> 3`; contact `< 1/2`; dormant means C stays below `0.1`. `g_lin`
  came from the local linear rate and was off by about `0.4` (on record).
- **Junction scale:** Q1 asks for a margin `> 10×` at radius 1 and `< 3×` at
  radius ≥ 2.

---

## Already audited elsewhere

The programme has swept several constants outside this stack. They are listed
so the ledger is in one place.

| constant | where swept | outcome |
|---|---|---|
| flow scale `t²E = 0.3` | `n3_kappa_deflation` | `κ ≈ 0.22` retired as a constant |
| fitting window, noise floor (area law) | `n3_gap_conventions` | survived; largest motion 38% of tolerance |
| deadband `0.01` (`trajectory_label`) | `n3_label_stability` | 2/3 held; a monotone dial |
| structured threshold `0.3` (Hopfield) | `n3_anatomy`, `n3_convention_manifold` | one knob; the consequence did not move on one substrate |
| `n_phases` guard | form-persistence backfill | replaced by `domain_diameter` |
| wall filters (formal half) | `reflection_wall_audit` | 2 of 6 walls mislabelled |

## Not covered

The reflection ladder, the gauge and Monte Carlo code, the scalar engine, and
the Hopfield and Kuramoto substrates. Each has its own constants, and the
sweeps above touch only some of them. This audit also doesn't re-check the
numerical results of the experiments it lists. It says which constants they
rest on.

## What could be done with this (not done here)

1. **Correct the stale 3-D text.** §6 of `The_Principle.md`,
   `Narrowing_the_N3_Question.md`, the `n3_selection_sweep` docstring and
   `Usage.md`. No new run is needed: the repo's own control table already
   explains the reading.
2. **Make 2-D-only measures refuse other dimensions.** For example, raise
   if `full_palette_junction_density` is called with `labels.ndim != 2` and no
   explicit valence, and check `dt·(4d + 2) < 2` in the step functions. That
   fixes the pattern rather than the instances, and the published 2-D case stays
   pinned.
3. **Re-run the six seeded experiments** with `× 2/P`, one at a time, quick mode
   first.
4. **Sweep `γ` and the step count** for the junction ranking in 2-D and 3-D.
   These are the two relaxation constants the headline result rests on that
   nobody has moved.
