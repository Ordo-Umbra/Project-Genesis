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

## The recurring failure type: regime-blind defaults

The same failure has now been found three times. A constant or default is
chosen in one setting and silently carried into settings it was never checked
against:

| constant | chosen for | broke in | found |
|---|---|---|---|
| junction threshold `m = 3` | 2-D vertices | 3-D and above (needs `m = d + 1`) | before 2026-10 |
| texture floor `domain_scale ≥ 2.5` | 2-D domains | 4-D (rejected resolved fields) | before 2026-10 |
| seed amplitude `[0, 0.1]` | (never checked) | everywhere: 40% of each field sits at `−1` | 2026-10-10 |

**Before trusting a new result, ask of every constant it rests on: in what
regime was this chosen, and does this run sit in it?** A full audit of the
measurement and relaxation constants is the top open item.

## Standing facts a session must know

- **Published seeding is defective.** `rng.random(...) * 0.1` conserves each
  sector at mean `0.05`, too little to tile space. About 40% of every relaxed
  field is then a sector at `−1`, labelled by `argmax` from residuals of order
  `0.01`. Use **volume-filling seeding** (`* 2/P`, mean `1/P`) for new work.
  The d+1 selection survives the fix and gets cleaner. Published *densities*
  should not be quoted. See `Docs/Junction_Selection.md`, "The seeding problem".
- **d+1 is geometric.** Random Voronoi selects it with no dynamics at all. What
  the relaxation contributes is genericity: degenerate junctions split.
  (`experiments/n3_junction_null.py`, `experiments/n3_junction_genericity.py`)
- **The d = 4 arms are weak.** Too few qualifying cells, and no matched Voronoi
  baseline fits a `24⁴` box. `d = 5` is beyond this instrument entirely.

## Setup and checks

```
pip install -r requirements.txt     # numba is required; scipy is not used
python -m unittest discover tests   # pytest is not installed
```

Experiments live in `experiments/` and run as plain scripts; most take
`--quick`. 4-D arms are opt-in (`--with-4d`) and take about an hour.

## Conventions

- Develop on the branch the session names, commit with descriptive messages,
  and push. Don't open PRs unless asked.
- Put model identifiers nowhere in the repo.
- Use `they/them` for anyone whose pronouns haven't been stated.
