# CLAUDE.md

Read this first, then the latest entries of `Docs/Session_Log.md`. Together
they are how a new session picks up where the last one stopped. Append to the
log before you finish; edit this file only when a principle or standing fact
changes.

## What this repo is

An executable testbench for the Universal Recursion Principle
(`S = ΔC + κ·ΔI`). It builds instruments, runs them, and reports verdicts,
including when the answer is no. Start at `README.md`. For the standalone
result most likely to be read by outsiders, see `Docs/Junction_Selection.md`.

## The collaboration

- The project lead (handle **OrdoIVOVI**) does all of this from a phone. Keep
  replies readable on a small screen: lead with the answer, use short tables,
  and don't make them scroll through raw output to find the point.
- The lead brings the larger view: structure emerging from simple distinctions,
  persistence as a filter, and understanding as an unending climb. They want
  pushback, not agreement. Keep their interpretation and the measurements
  clearly apart. Say which claims are **measured**, which are **consistent
  with** the measurements, and which the measurements **don't speak to**.
- **Pace matters as much as content.** The lead has said the speed at which
  results arrive is hard to keep up with. Do one thing per session unless
  asked for more. After it, stop and give a short summary: what changed, what
  it means, and the one thing worth deciding next. Don't chain into the next
  experiment without being asked. When a finding is large (one that changes
  what earlier results mean), say so plainly and let it settle before building
  on it.
- Ups and downs are expected. What matters is that each failure is specific,
  recorded, and traced to a cause. **The same type of failure recurring means
  something systemic is wrong**: look for the pattern, not just the instance.

## Working principles

1. **Pre-register.** Every experiment's docstring states its predictions and
   falsifiers before it runs, and the script prints a verdict for each one.
2. **Failed verdicts stay failed.** If a prediction fails, report it as failed
   and explain why. Diagnostics added after a run are labelled *post hoc* in
   both the script output and the doc.
3. **Pin before you extend.** A new or generalised measure must reproduce the
   published one exactly in its default case, with a test that proves it (see
   `tests/test_junction_scale.py` and `tests/test_wetting.py`).
4. **Null models before interpretation.** Ask what a structureless baseline
   (for example random Voronoi) scores before crediting the dynamics.
5. **Quote rankings over magnitudes** when sample counts are small, and say
   how small.
6. **Keep the record; keep the core current.** This is the lead's rule
   (2026-10-31), and the next section says how to apply it.

## Record and core documents

Two kinds of document, handled differently.

**Record** is what was run and what was found, as it was found. It is never
rewritten or deleted. If a later result changes what an entry means, add a
short dated note at that entry pointing to the correction, and leave the entry
itself alone. Record documents:

- the dated entries of `Docs/Experiment_Log.md`
- `Docs/Session_Log.md`
- dated working notes, such as `Docs/Narrowing_the_N3_Question.md`
- experiment docstrings: their predictions and post-run notes
- the other essays in `Docs/`, until the lead says otherwise

**Core** is what the programme holds now. It is updated in place as results
change, so a reader gets the current position without reading the history.
Where a claim changed, say so briefly and point to the record. Core documents:

- `README.md`
- `Docs/The_Principle.md`
- `Docs/Junction_Selection.md`
- `Docs/Usage.md`
- `Docs/Constants_Audit.md` (statuses update as fixes land; findings stay listed)
- the summary table at the top of `Docs/Experiment_Log.md`
- this file

New core documents will appear as the work grows. Add each one to this list
when it does. When a correction lands, check every core document for the old
claim, not just the one where it was noticed.

## The recurring failure type: regime-blind defaults

The same failure has now been found four times, and a fifth has been derived
but not yet observed. A constant or default is chosen in one setting and
silently carried into settings it was never checked against:

| constant | chosen for | broke in | found |
|---|---|---|---|
| junction threshold `m = 3` | 2-D vertices | 3-D and above (needs `m = d + 1`) | before 2026-10 |
| texture floor `domain_scale ≥ 2.5` | 2-D domains | 4-D (rejected resolved fields) | before 2026-10 |
| seed amplitude `[0, 0.1]` | (never checked) | everywhere: 40% of each field sits at `−1` | 2026-10-10 |
| κ recovery default `r = 0.1` | (unrecorded) | 2-D recall and 3-D form persistence: starves the field | before 2026-10 |
| time step `dt = 0.1` | 2-D | at risk in 4-D, unstable in 5-D (derived, not observed) | 2026-10-31 |

**Before trusting a new result, ask of every constant it rests on: in what
regime was this chosen, and does this run sit in it?** The audit of the
multiphase field stack is `Docs/Constants_Audit.md` (2026-10-31). Read it
before trusting a field result. Its main finding about the pattern: every fix so
far was added as an opt-in, with the old default left in place, so fixed
defects stay live for callers who don't opt in, and older docs still quote them.

## Standing facts a session must know

- **Published seeding is defective.** `rng.random(...) * 0.1` conserves each
  sector at mean `0.05`, too little to tile space. About 40% of every relaxed
  field is then a sector at `−1`, labelled by `argmax` from residuals of order
  `0.01`. Use **volume-filling seeding** (`* 2/P`, mean `1/P`) for new work.
  The d+1 selection survives the fix and gets cleaner. Published *densities*
  should not be quoted. See `Docs/Junction_Selection.md`, "The seeding problem".
  Six experiments use the defective seeding with conserved dynamics; the list
  is in `Docs/Constants_Audit.md`.
- **The 2-D junction threshold `m = 3` is still the default** in
  `multiphase.full_palette_junction_density`, `count_triple_junctions` and
  `selection.py`. In 3-D it reads edges, not vertices, so `P = 3` wins there by
  counting alone. The core docs were corrected on 2026-10-31; the record still
  shows the old reading, with dated notes. For any `d`, use
  `junction_scale.full_palette_density(min_valence=d+1)`.
- **d+1 is geometric.** Random Voronoi selects it with no dynamics at all. What
  the relaxation contributes is genericity: degenerate junctions split.
  (`experiments/n3_junction_null.py`, `experiments/n3_junction_genericity.py`)
- **The d = 4 arms are weak.** Too few qualifying cells, and no matched Voronoi
  baseline fits a `24⁴` box. `d = 5` is beyond this instrument entirely.

## Setup and checks

```
pip install -r requirements.txt     # numba is required; scipy is used by count_domains and a few experiments
python -m unittest discover tests   # pytest is not installed
```

Experiments live in `experiments/` and run as plain scripts; most take
`--quick`. 4-D arms are opt-in (`--with-4d`) and take about an hour.

## Conventions

- Develop on the branch the session names, commit with descriptive messages,
  and push. Don't open PRs unless asked.
- Put model identifiers nowhere in the repo.
- Use `they/them` for anyone whose pronouns haven't been stated.
