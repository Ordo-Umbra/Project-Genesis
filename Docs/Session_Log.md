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
