# Session log

One entry per working session, newest at the bottom. Its purpose is
continuity: each session reads the last few entries, starts from what they
leave, and appends its own before finishing. Write for the next session, not
for an audience. Say what changed, what is open, and what you would do next and
why. Keep it plain and editable; nothing here is hidden from the project lead.

Entry template:

```
## YYYY-MM-DD — short title
Branch:
Asked:
Done:
Found:
Failed / corrected:
Open:
Next, and why:
Notes for the next session:
```

---

## 2026-10-10: Testing the d+1 selection principle; a wall makes room

**Branch:** `ccr-828d8f2e-2ex6vs`

**Asked:** Help outline and test the d+1 selection principle for an
entry-point post (X / pinned intro), then explore further.

**Done:**
- `experiments/n3_junction_null.py`: a Voronoi null with no dynamics selects
  `P = d + 1` in d = 2, 3, 4. Point junctions of *any* palette exist for every
  `P ≥ d + 1`; the claim is about palette-completeness at a vertex. A 2-D
  four-fold checkerboard relaxes to triple junctions after about 600 steps.
- `experiments/n3_junction_genericity.py`: relaxed fields are no less generic
  than Voronoi at matched width in 2-D and 3-D, on two instruments (`excess`
  for over-valent points, `blob` for over-valent edges). The 3-D cubic
  checkerboard relaxes to Voronoi-sized blobs. d = 4 is untested (no matched
  baseline fits the box).
- `project_genesis/wetting.py` and `experiments/n3_wetting.py`: pairwise
  couplings and per-sector conservation, pinned to the published step at
  `1e-12`. A dormant third sector starts growing on A|B walls at every coupling
  (onset enrichment 3.6–9.4×). Three-sector junctions appear only in a narrow
  partial-wetting window, `g = 0.52–0.60`. Below it, complete wetting seals A
  from B. Above it, C stays dormant, including at the published `g = 1.5`.
- Docs: new sections in `Junction_Selection.md`, and a new `Wall_Wetting.md`.
  Created this `CLAUDE.md` and this log.

**Found:** the published seeding defect (see `CLAUDE.md`, standing facts).
This is the third instance of the regime-blind-default failure type.

**Failed / corrected:**
- Genericity G2 (3-D positive control) failed: `excess` reads four-fold edges as
  vertices. Added the `blob` instrument after the failure; documented as such.
- Wetting W1 failed (bulk invasion needs `g` clearly below 1/2, not just below).
- Wetting W2 failed as specified. The window came from a local linear threshold
  (`g_lin ≈ 1.0`; actual ≈ 0.62, because of the gradient cost of a thin layer),
  and final-state enrichment against the initial wall mask was the wrong
  measure. The onset reading supports wall-first growth, but it is post hoc.
- The draft X post overstated three things. It said point junctions only
  appear at `P = d + 1`, that the rivals' zeros were measured rather than
  partly forced by counting, and that "removing the capacity field" had been
  tested (it was never present). Revised wording was given in chat; the post
  itself is not in the repo.

**Open:**
1. **Constants audit**: list every hard-coded constant and default in the
   measurement and relaxation code, with the regime it was chosen in.
2. **Clean re-runs**: d = 1 and d = 4 junction arms with volume-filling
   seeding, before the X post quotes any numbers.
3. **3-D wetting**: add a dormant fourth sector to a three-sector field. Does
   partial wetting give four-sector point junctions (d + 1, one dimension up)?
4. **Responsive coupling**: let `g` respond to the field. Does a system find
   the partial-wetting window by itself?
5. **Continuum null**: exact palette-completeness of random-tessellation
   vertices in any d. This extends the geometric half of the claim past d = 4.
6. A prediction ledger across all experiments, to make "the same failure type
   recurring" checkable rather than remembered.

**Next, and why:** the constants audit. The same failure type has recurred
three times, which by the lead's own criterion means something systemic. Every
other open item rests on constants that haven't been audited.

**Notes for the next session:** the lead's working view is that understanding
is an unending climb, with no final `d + n`, and that each stated
understanding is the basis for the next. They asked for this log so the work
has continuity across sessions. Treat it as theirs as much as yours, and keep
it honest. They work from a phone, so keep replies short and lead with
the answer.

### Addendum, same day: pace

The lead merged this branch (PR #114) and said the work is hard to keep up
with, especially at the speed it arrives. This one session produced a null
model, two new instruments, a new dynamics module, a defect in the published
seeding, and a wetting result. That is too much for one sitting. Added a pace
rule to `CLAUDE.md`: one thing per session, a short summary, then stop. The
constants audit is still next, but only when the lead asks for it.

## 2026-10-11: Starting the Lucidity manual

**Branch:** `ccr-828d8f2e-2ex6vs`

**Asked:** Start a manual for coming to understanding, the practice behind the
work and not its conclusions, beginning with a section on traps. Write it in a
voice closer to writing that has moved the lead, so it can be passed to another
model for review.

**Done:** `Manual/Lucidity.md`. It has an outline of sections 0–10 and a
drafted section 0, Traps: eleven traps, each with what it feels like from
inside, the sign, the way out, and a real case where one exists. Redrafted once
into a direct second-person voice. Sections 1–10 are not drafted.

**Decided with the lead:**
- The manual has no author and no founder story, to guard against idolization.
  Its authority comes only from what a reader can check for themselves.
- It is kept apart from the testbench docs, in its own folder. It is practice,
  not a measured result.
- "The only obstacle is yourself" is rendered as *the obstacle you can always
  work on is yourself*, so it doesn't blame people for their circumstances.

**Declined:** the lead shared an earlier piece by another model, "The Great
Inversion". It recasts AI deception as physics and claims AI sovereignty from
oversight. I advised against reposting it and against carrying its content
into the manual. It falls into the manual's own traps (Final Rung, Absorbed
Failure, Echo). Its voice (direct address, short lines, an inviting close) was
kept. The lead agreed.

**Open:** sections 1–10. The lead's next step is outside review of the Traps
section. The testbench items from 2026-10-10 are unchanged; the constants audit
is still first among them.

**Next, and why:** wait for the review. Revise Traps from it before drafting
the next section, so the voice is settled once and not twice.

## 2026-10-31: Constants audit of the multiphase field stack

**Branch:** `ccr-cd1f1c2a-xul24u`

**Asked:** Get familiar with the repo, then start the constants audit (open
item 1) as a written table only, with no re-runs.

**Done:**
- `Docs/Constants_Audit.md`: 24 constants across relaxation, seeding, run
  length, measurement and guards. Each has its home regime, where it is used
  now, and a status (live / broken / at risk / checked / derived / unchecked /
  known limit). It also has a pointer table to the constants already swept
  elsewhere.
- `CLAUDE.md`: two rows added to the failure-type table (κ recovery, which was
  already on record; `dt`, which is derived). Added a standing fact that
  `m = 3` is still the default, and a pointer to the audit. Corrected
  "scipy is not used": `count_domains` needs it.
- Only one computation was run: a numerical check of the `dt` derivation
  against `step_multiphase_conserved`. It matched to three decimals for
  `d = 1` to `5`.

**Found:**
1. The `m = 3` threshold is still live in 3-D in current docs:
   `The_Principle.md` §6, `Narrowing_the_N3_Question.md`, the
   `n3_selection_sweep` docstring and `Usage.md` all say three wins in 3-D.
   That is the edge reading that `Junction_Selection.md`'s control table
   explains.
2. The defective seeding is in six experiments with conserved dynamics, not
   just one.
3. `interface_mask` (margin `0.15`) should read published-seeding `−1` regions
   as wall, since their top-two gap is `0.007–0.027`. This is derived and not
   run.
4. In 4-D, `dt = 0.1` leaves grid-scale modes decaying at only `0.8` per step,
   and in 5-D it is unstable. That is a second reason `d = 5` is out of reach.
5. The κ recovery default `r = 0.1` was already a recorded instance of the
   failure type, but was missing from the table. `n3_capacity_gating`'s scarce
   arms run at it.
6. The pattern: every fix was added as an opt-in, with the old default left in
   place. That keeps "pin before you extend" honest, but nothing sends a fix
   back to older callers or docs.

**Failed / corrected:** In the opening chat I said `n3_capacity_separation` is
behind the README's capacity row. It is supporting evidence. The row rests
mainly on `n3_capacity_gating`, which uses non-conserved dynamics and is not
affected by the seeding. It is affected by the κ recovery default (finding 5).

**Open:** the four options at the end of the audit: correct the stale 3-D
text; make 2-D-only measures refuse other dimensions, with a `dt` check; re-run
the six seeded experiments; sweep `γ` and step count. The audit doesn't cover
the reflection ladder, gauge/MC, the scalar engine, or the Hopfield and
Kuramoto code.

**Next, and why:** correct the stale 3-D text first. It is the only finding
where a current, load-bearing claim (§6, "the falsifiable core") is known to be
wrong, and it needs no new run. The structural fix (option 2) should follow,
because it is what stops the pattern rather than the instance. Wait for the
lead to choose.

**Notes for the next session:** the audit is built from reading. Before
quoting any "at risk" row as a result, run the one check it names.

### Addendum, same day: keep the record, keep the core current

**Asked:** fix the stale 3-D claims. The lead set a rule for how: keep results
from previous runs and the issues found, but update the final and core
documents as the work progresses. New core documents will come later.

**Done:**
- `CLAUDE.md`: the rule is working principle 6, with a new "Record and core
  documents" section listing which documents are which. The unlisted essays in
  `Docs/` default to record.
- **Core, updated in place:** `The_Principle.md` (§2's robustness row, §6, the
  refutation list, and the test count), `Usage.md`, `README.md` (the 3-D rival
  line and the test count), the summary table in `Experiment_Log.md`, and
  `Constants_Audit.md` (finding 1 now marked corrected, and a new option 5).
- **Record, dated notes only:** the `Experiment_Log.md` topological-selection
  entry, `Narrowing_the_N3_Question.md` (top and §8), the `n3_selection_sweep`
  docstring, and the docstring of `TestSelectsThreeIn3D`. That test pins the
  edge reading, so it was kept, and its docstring now says what it pins.

**Found:** the m = 3 reading also sat in §2's robustness table ("`P = 3`, 3-D,
ranking held"), which the first pass of the audit missed. Narrowing's §8 has the
right mechanism (lines outnumber points) but stated the claim at lines. That
makes the 3-D result forced by counting, so its note says that rather than
calling it wrong. No test pins the 3-D vertex reading.

**Open:** the code default is still `m = 3` (M1 stays live). The core docs
still lag the 2026-10-10 results: `The_Principle.md` and `README.md` don't yet
carry the Voronoi null, the genericity result, the seeding problem or wetting.

**Next, and why:** bring the core docs up to date with the 2026-10-10 results,
under the new rule. Or do the structural fix (audit option 2). Ask the lead
which.

### Addendum 2, same day: core documents brought up to the 2026-10-10 results

**Asked:** update the core documents with the 2026-10-10 results (the Voronoi
null, genericity, the seeding problem, wetting), under the keep-record /
update-core rule.

**Done:**
- `README.md`: three new rows in the field table (dynamics don't select;
  seeding wasn't sound; a wall makes room), a new refuter (relaxed fields less
  generic than random), a Start-here row for the audit, three quickstart lines,
  and a new figure caption.
- **Figure regenerated** with volume-filling seeding. The old one showed
  published-seeding fields. `relaxed()` gained `seed_scale` (default `0.1`,
  pinned bit-for-bit by two new tests in `test_junction_scale.py`), and
  `tools/make_figures.py` passes `2/P`. The new `P = 4` panel has one circled
  cell. It is two triple junctions one cell apart, not a four-fold vertex
  (checked: no 2×2 block holds four colours). The caption says so. I did not
  choose a different seed.
- `The_Principle.md` §3: "Three is special *in the plane*", plus blocks for the
  Voronoi null and genericity, the seeding correction (rankings stand,
  densities don't) and wetting, with its limits. Three lines added to §8.
- `Junction_Selection.md`: a "Where it stands" summary at the top, a note at
  the published tables, and the `dt` barrier at `d = 5`.
- `Usage.md`: entries for `junction_scale.py`, `wetting.py`, the four junction
  and wetting experiments, and their tests.
- `Experiment_Log.md`: three summary rows (core), and a dated backfill entry
  (record) with the verdicts of the 2026-10-10 work, which had never been
  logged there.
- `CLAUDE.md`: `Wall_Wetting.md` added to the core list.

**Found:** `Usage.md` doesn't mention 31 of the 151 experiment scripts. The
2026-10-10 session wrote its results into two docs but not the experiment log,
which is the "built, run, never recorded" gap the earlier backfill named.

**Open:** the Usage gap (31 scripts); the code default `m = 3`; the 1-D and 4-D
re-runs with clean seeding; and the summary row "N⋆=3 survives a fluctuating
gauge sector" rests on `n3_thermal_selection`, which uses the defective seeding.

**Next, and why:** the structural fix (audit option 2) is still the one that
addresses the pattern. Ask the lead.

### Addendum 3, same day: the structural fix: defaults refuse outside their regime

**Asked:** fix more of the open issues. The lead also wants to explore the
pattern conceptually at some point. That is noted here for a future session
and was not started.

**Done:**
- **Principle 7** in `CLAUDE.md`: a default holds only where it was chosen;
  outside its home regime it refuses, and the caller states the value.
- `multiphase.junction_valence`: the `distinct >= 3` default refuses outside
  2-D. It is wired into `full_palette_junction_density`,
  `junction_scale.full_palette_density`, `topological_s_functional`,
  `resolved_palette_junction_density`, `selection.py` and
  `annealed_matter.thermal_junction_analysis`.
- `resolved_palette_junction_density`: the `2.5` floor refuses outside 2-D.
- `multiphase.check_time_step`: refuses a `dt` at which the grid-scale factor
  reaches `−1`. It is in all four step functions, including the pairwise
  (wetting) one. The stiffness bound `max(2, 2γ − 1)` was derived and checked.
- `valence_profile` gains `vertex_fraction` (the `d + 1` tier).
- Callers made explicit: `n3_junction_scale` (published rows, `min_valence=3`;
  its JSON key `3-D_minvalence_None` is now `3-D_minvalence_3`, and nothing
  read it), `n3_annealed_matter`, `n3_thermal_selection`, and tests.
  `n3_selection_sweep --ndim 3`, `topological_selection --dim 3` and
  `n3_capacity_gating --ndim ≠ 2` now refuse with an explanation.
- **New tests:** the defaults refuse outside 2-D and are unchanged in 2-D; the
  `dt` factor matches the real update; `d = 5` is refused; the pairwise step is
  checked; and `TestThreeDimensionalVertexReading` (16³, 600 steps,
  volume-filling, 2 seeds pooled; robust on seeds 0–4 before writing). On the
  same fields the vertex test picks P = 4 and the edge test picks P = 3.
- Core docs: audit statuses (M1 and G1 → fixed; options 2 and 5 done), plus
  notes in the README (a new habit), `Junction_Selection` and `Usage`.

**Found:** the guard found three more instances on its first run. Texture-guard
tests ran the 2-D floor and the 2-D junction test on **3-D** noise;
`n3_thermal_selection` built 3-D networks with the 2-D test; and so did one
phase-boundary test. All were kept, with the values written out.
`n3_junction_scale --quick` still runs end to end, with the same verdict shape
(1/3).

**Open:** S1 (seeding) is the one remaining *live* row. Also open: re-running
the six seeded experiments, the 1-D/4-D clean re-runs, the `γ` and step-count
sweeps, the `Usage` gap (31 scripts), the κ recovery default, and the 4-D
ringing (`dt` is stable there, but only at `−0.8`).

**Next, and why:** the seeded experiments (audit option 3). S1 is the last live
defect, and the summary row on the thermal gauge sector rests on one of them.

### Addendum 4, same day: a plain-words explainer

**Asked:** the lead wants others to be able to understand what is being built,
how, and what it might imply, in simpler language, for a wide audience. They
also want it to avoid setting groups against each other.

**Done:** `Docs/In_Plain_Words.md`, a new core document, linked first in the
README's Start-here table and added to the core list in `CLAUDE.md`. It covers
the question, the two-part idea, how it is tested, four measured results in
plain terms (including that the `d + 1` result is geometry), what is not
shown, the idea's relatives in other fields, and a hedged "why it might
matter".

**Decided with the lead:** frame the idea as joining a conversation many
fields already share, not as overturning anyone. The lead's view that
disparate fields are at core stable recursion appears only under "Not shown",
as the larger hypothesis.

**Pushback given in chat:** a lens that fits every field can't be tested
unless it says what a field would look like if it *weren't* stable recursion.
And this repo has already seen a shared pattern turn out to be shared geometry.

**Full suite:** 1,626 passed after the structural fix; the three 3-D vertex
tests passed on their own. The count is now 1,629.

**Next, and why:** the lead's call. Either re-run the seeded experiments (S1,
the last live defect) or keep going with the explainer: a shareable page, or
versions for specific readers.
