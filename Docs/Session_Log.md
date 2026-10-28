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

## 2026-10-28: Reading the repo cold, and re-running what it rests on

**Branch:** `ccr-27e79fb1-ajcm6l`

**Asked:** First session in this setting. Read all the findings, run checks
where useful, then ask questions. No experiment was requested, and none was
added to the repo.

**Done:**
- Read `README.md`, this log, `The_Principle.md`, `Junction_Selection.md`,
  `Wall_Wetting.md`, the structure and closing sections of
  `The_Reflection_Ladder.md`, the summary, backfill and roadmap of
  `Experiment_Log.md`, `Manual/Lucidity.md`, and the opening of the founding
  URP text.
- Re-ran the full suite: `1615` tests, all pass, `713 s`. Re-ran `n3_junction_null.py`: all seven
  verdicts and every number reproduce exactly. Re-ran `reflection_ladder.py`:
  `11,103,120` against `4,996`, the 2,222x, in one second.
- Independent checks, from scratch scripts that are not committed. Recipes:
  `n3_junction_scale.relaxed` with `* 2/P` in place of `* 0.1`, `γ = 1.5`,
  `dt = 0.1`, seeds `71 + 100P + t` unless stated.

| check | doc says | measured here |
|---|---|---|
| published seed, 2-D `P=3`, `m=3`, 72², 600 steps, 3 trials | `0.00694` | `0.00694` |
| published seed, 2-D `P=3`, cells only at `−1` | `0.415` | `0.422` |
| volume-filling, 2-D `P=3`, same protocol | `0.01530` | `0.00945` |
| same, 10 seeds (0–9) | — | mean `0.0089`, range `0.0064–0.0131` |
| same, 10 seeds, 1500 steps | — | mean `0.0056` |
| volume-filling, 3-D `P=4`, `m=4`, 28³, 1500 steps, 2 trials | `0.00271` | `0.00481` |
| volume-filling rankings, 2-D and 3-D | `d + 1` alone | `d + 1` alone, every rival `0` |

**Found:**
1. **The volume-filling densities in "The seeding problem" don't reproduce,
   and no committed script produces them.** The `0.01530` sits above all ten
   seeds I ran. The rankings reproduce exactly. Density also falls with run
   length, because the fields keep coarsening, so a density is partly a
   statement about when the run stopped. Quote rankings only.
2. **Post hoc: the one-dimension gap appears on a structureless field.**
   `n3_area_law`'s own Gaussian control, read with its own instrument
   (`n = 160`, halves `22–44`, 48 seeds, seed 0), gives `n_C = 2.01 / 1.97 /
   1.98` and a bias-corrected `n_I = 1.02 / 0.98 / 1.00` at `ξ = 1.5 / 3 / 6`.
   That is a gap of `1.00 / 0.98 / 0.99`, invariant under a 4x change in
   correlation length. Raw gaps: `0.86–0.88` on the control against `0.92` on
   the field. Caveat: the `+0.12` bias correction is calibrated on this same
   control, so `n_I ≈ 1` there is partly by construction; `n_C ≈ 2` is not.
   Measured: the field has the gap. Consistent with: URP's claim that the
   asymmetry is universal. Doesn't speak to: whether URP's dynamics produce
   it, since locality and a finite correlation length are enough.
3. **The front doors predate the October work.** `README.md` and
   `The_Principle.md` mention neither the Voronoi null nor the seeding defect.
   `The_Principle.md` §6 still calls the `P = 3` selection "the falsifiable
   core" and "what is left standing", which a random Voronoi tessellation
   also passes.
4. Test counts have drifted: README `1599`, `The_Principle.md` `1255`, CI
   comment `1184`. There are `1615` test functions.

**A second recurring failure type.** Alongside regime-blind defaults:
*crediting the principle with something a structureless baseline also
does.* Seen twice now, for both of `The_Principle.md` §2's "two independent
measurements": the cliff (d+1, which Voronoi passes) and the gap (which a
Gaussian field passes). Principle 4 in `CLAUDE.md`, null models first, was
written after most `[measured]` claims existed. Only d+1 has been
retro-nulled.

**Open (put to the lead):**
1. What does URP predict that a generic field does not? §5's eviction
   condition and the continual-learning benchmark are the candidates I have
   not null-checked.
2. "The number of generations is `d + 1`, four in three" (`The_Principle.md`
   §3, tagged `[measured]`). Our world is 3-D with three generations. A fourth
   with a light neutrino is excluded by the LEP Z width (`N_ν = 2.984 ±
   0.008`). Is "generations" a metaphor or a prediction?
3. The flat-cost bridge (README's "next experiment") gives as its falsifier
   "the surface-law exponent moves when the addressing convention changes".
   The lattice field has no addressing convention. Given item 2 of Found, is
   the bridge still worth building?
4. Has the X post gone out, and does it quote any density?
5. Has the Traps section had its outside review?

**Next, and why:** the lead decides. My recommendation is a null audit: take
each `[measured]` claim in `The_Principle.md` that is credited to URP and ask
what a structureless baseline scores. It decides what the front doors should
say, so it should come before rewriting them, and it bears more on the
programme than the constants audit does. A result that any generic field
produces doesn't depend on its constants for its meaning.

**Notes for the next session:** none of the above changed a file other than
this log. The scratch checks took under two minutes in total. If any becomes
a claim, it needs a pre-registered experiment file first.

### Addendum, same day: the compass

The lead shared "A Compass for the Third Thing", a short piece on practice
(three movements: poles, play, emergence; then a four-step practice). At the
lead's request it is now at `Manual/A_Compass_for_the_Third_Thing.md`, beside
Lucidity, in their wording (only heading marks added). It is practice, not a
result. Read against the measurements:

- **Consistent:** "don't choose a pole" matches the wetting window, where
  three-sector junctions exist only between C sealing every wall and C never
  switching on. "Passed between forms" matches §5: the condition transplants,
  the route doesn't.
- **Contradicted unless qualified:** "make one thing better, then the next;
  the rest builds itself". The ladder's §8e and §8f are counterexamples:
  local productivity forever with no advance. The qualifier the ladder
  supports is flat reference cost (§3, §8h).
- **Not spoken to:** purpose arising, and treating the other like yourself.
- **Testable:** whether a system finds the in-between by itself. This is open
  item 4 from 2026-10-10 (responsive coupling).

### Addendum, same day: who sets the dial

From discussing purpose and capacity with the lead: in each place this repo
finds structure appearing, the dial that lets it appear was set by hand. The
wetting window (`g = 0.52–0.60`) was swept, not found. Self-repair under
κ-gating needs `r = 0.8`, while the default `r = 0.1` starves the field before
anything forms. No instrument here lets a system set its own dial. "Can a
system find the setting that lets it build?" is the shared untested question
behind open item 4 (responsive coupling), and the lead names it as the
current problem.
