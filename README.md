# Project Genesis

**An executable testbench for the Universal Recursion Principle.** The theory
proposes that recursive systems — physical, biological, cognitive — evolve by
climbing a single scalar:

```
S  =  ΔC  +  κ · ΔI
```

**ΔC** is *distinction*: making differences, articulating structure. **ΔI** is
*integration*: binding those differences into something coherent. **κ** is the
finite *capacity* that decides how much integration a system can afford.

The claim that does the work is an asymmetry. Distinction is cheap — noise makes
gradients. Integration is expensive, and the bill comes due exactly where
distinction is richest. So a recursive system can always distinguish more than it
can integrate, never catches up, and *in failing to catch up it builds
structure*.

**This repository does not argue for that.** It builds instruments that measure
it, runs them, and reports verdicts — including when the answer is no.

---

## Start here

| If you want | Read |
|---|---|
| The theory, from zero, in half an hour | [`Docs/The_Principle.md`](Docs/The_Principle.md) |
| One result that stands entirely alone, no framework needed | [`Docs/Junction_Selection.md`](Docs/Junction_Selection.md) |
| The full chronological record, ~100 experiments with verdicts | [`Docs/Experiment_Log.md`](Docs/Experiment_Log.md) |
| The reflection-ladder results in plain language, no proof theory assumed | [`Docs/The_Reflection_Ladder.md`](Docs/The_Reflection_Ladder.md) |
| To actually run the thing — CLI, agents, server, browser toys | [`Docs/Usage.md`](Docs/Usage.md) |

Every claim in `The_Principle.md` is tagged with its standing: **[measured]** —
a pre-registered experiment with a verdict; **[framework]** — a coherent
consequence that has *not* been tested; **[declined]** — something the programme
explicitly refuses to claim.

---

## What it looks like

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="Docs/img/junctions-dark.png">
  <img alt="Four relaxed multiphase sector fields at palette sizes 2, 3, 4 and 5. Only the three-sector field contains cells whose neighbourhood carries every sector at a vertex; these are circled. The others contain none." src="Docs/img/junctions-light.png">
</picture>

Four relaxed sector fields. A circled cell is one whose immediate neighbourhood
carries **every** sector at once, at a vertex rather than along a wall. Two
sectors cannot form such a junction; four and five form plenty of junctions but
essentially none that bind the whole palette. Three can, and does.

That picture is the shape of the central result, and it generalises: in `d`
dimensions the palette that maximises full-palette junction density is **`d + 1`**
— measured at `d = 1, 2, 3, 4`. ([the standalone note](Docs/Junction_Selection.md))

---

## What has been measured

Load-bearing results only. Negatives included, because they are the point.

The programme runs on two instruments. The **field** half puts the principle on
a lattice and measures it; the **formal** half puts it on a reflection ladder —
towers of theories, each proving the consistency of the one below — where the
same asymmetry can be stated without a lattice at all. They meet at exactly one
*measured* point, and that is said plainly further down.

### The field

| Question | Verdict |
|---|---|
| Does a field select a palette size, and which? | **`d + 1`**, measured at `d = 1, 2, 3, 4`. Rivals are *identically zero* in 1-, 2- and 4-D; in 3-D the nearest is live at `10×` behind |
| Is that selection driven by capacity scarcity? | **No — refuted.** It is geometric: codimension counting, of which Plateau's trivalent vertex is the 2-D case. `P = 3` peaks at every capacity level *including no capacity field at all* |
| Does distinction outrun integration by a fixed amount? | **Yes, in 2-D.** Distinction is volume-law (`n = 1.99`), integration is surface-law (`n = 0.95`) — a gap of `1.04`, one dimension, and the worst motion under any convention swept is 38% of the experiment's own tolerance |
| Does the capacity law transplant off the lattice? | **The eviction condition does** — to a Hopfield network and a Kuramoto oscillator population, three structurally unrelated substrates. Whether the crossing sits at a substrate-independent `κ` is **not settled** |
| Is the optimum's *route* to criticality a substrate fact? | **No.** It moves when you change how ΔC is read. The *condition* survives; the route was measuring the dictionary |
| Does scarcity evict an ordered optimum? | **Yes, ceiling-free**: exactly when the ordered point starts ahead and holding it costs capacity |
| Does capacity hold structure together, or only ration it? | **It holds it — above a recovery threshold, and the default sat below one.** A noise-driven junction network self-repairs with system size only under κ-gating, and only where capacity regenerates fast enough (`r = 0.8`): `⟨κ⟩` and domain width both *rise* with the box. At the default rate the same dynamics starve the field before any structure forms |
| Have the robustness claims themselves been checked? | **Yes, and `2/3` of the re-scoring predictions failed.** One claim — §2's gap — turned out *unevidenced by all three sweeps cited for it*; the others were genuinely tested and held |

That last row is the honest headline. A claim can pass a robustness bar because
nothing moved rather than because it resisted, and telling those apart needed
[a separate instrument](project_genesis/robustness.py). Applied retrospectively
it refuted two of its own three predictions — the blind spot turned out to be
narrower than feared, and the `P = 3` selection survived a convention that
genuinely threatened it.

### The reflection ladder

The formal twin of the capacity gap: `T₀ = PA`, `T_{n+1} = T_n + Con(T_n)`, a
tower that never tops out. `The_Generative_Gap.md` had cited it for a long time
without running it. Running it needed one methodological move, because two of
the three quantities involved are *definitions* — `G > 0` holds because the next
rung was constructed, so a program reporting it would be a tautology with a
progress bar. So every experiment here is built around a quantity that can come
out other than the model says, each with a deliberately broken control arm that
makes it do so.

| Question | Verdict |
|---|---|
| Does climbing the ladder buy capability, or symbols? | **Naming decides, and the mathematics does not fix the naming.** Two towers, floors one-for-one identical in what they add: `11,103,120` symbols against `4,996` at floor 12 — a factor of `2,222`. The broken arm's address counter wraps: the floor number climbs to 11 while the axiom set stays frozen at 16 |
| Do the constraints that make continuation meaningful protect *advance*? | **No — they select against it**, 9/9 rows. But not because they are constraints: because they read the size of the thing reflected on, and advancing always reflects on the largest object. Price by *description* instead and all 9 go neutral |
| Does breadth convert into height? | **No. Four times the options bought exactly the same 80 rungs.** What breadth buys is the worst case — concentration wins the mean, diversification wins the floor, and which line fails is precisely what the interior cannot see |
| Can a system tell, from inside, whether its progress still counts? | **No, and not for want of attention — there is no signal.** A retracted foundation and an unaffordable next step produce records identical attempt for attempt. One probe does work: re-derive your own foundation, which is the cheapest and shortest-named thing you have, so a refusal there is neither price nor encoding. It is conditional, and the condition is measured |
| Was the wall taxonomy itself checked? | **Yes, and two of six walls were mislabelled.** Perturbing the price and the numbering separately shows that the "epistemic" filter was a size tax with a label, and the *structural* wall reads partly how the nodes happen to be numbered |

The arc matters more than any row of it. Six proxies for progress were measured
and each was then dissociated from what it stood in for: more representation,
more steps, continued motion, local productivity, more rank, and finally *your
own measurement*. The last is the one that unsettles the others — every earlier
proxy fails against an external check, and that one is the external check
failing.

One mechanism survives all of it: **description-addressing**, where the cost of
referring to something does not scale with the thing referred to. Flat cost is
what makes the walls non-directional, which is what removes the pull toward
going sideways, which is what lets a climb continue without bound.

### Where the two halves meet

At one measured point: [the functor experiment](Docs/Experiment_Log.md#the-functor-logic--vacuum-measured), which
found the vacuum's topological content to be a path-independent function of the
integration level — well-defined on objects, and contravariant, with the
direction explained. Everything else connecting the halves is structural
resemblance, and it is not evidence.

There is a second connection that is *not yet measured*, and it is the sharpest
open question here. `inline` pricing doubles with what it refers to; `indexed`
stays flat however much theory it names. Distinction is volume-law (`n = 1.99`);
integration is surface-law (`n = 0.95`). Both say the same thing — cost that
does not scale with the thing referred to is what buys unbounded structure —
and the formal half has just spent twelve experiments establishing that this
exact property is the one that survives every dissociation. Whether those are
one mechanism or a pun has not been tested. It is the next experiment.

## What would refute it

- A palette other than `d + 1` scoring higher on a resolved field, in any dimension.
- A capacity sweep that moves the integrated fraction (run — it does not).
- An integration measure that scales with volume rather than surface.
- A cost model that is flat in the reflected object and *still* selects against advance — the chain in `§8h` of the ladder document is a chain, so any link breaking breaks it.
- An interior probe that separates a retracted foundation from an unaffordable step without the condition `§8k` attaches to it.

Those are the measurements the rest depends on. Everything else is consequence.

---

## Quickstart

```bash
pip install -r requirements.txt

# the central result, d = 1, 2, 3
python experiments/n3_junction_scale.py
python experiments/n3_junction_scale.py --with-4d   # adds d = 4 (~60 min)

# the scaling dimension of the gap, and its convention audit
python experiments/n3_area_law.py
python experiments/n3_gap_conventions.py

# the capacity law on three substrates
python experiments/n3_criticality_transplant.py
python experiments/n3_kuramoto_transplant.py

# the formal half — seconds each, no lattice
python experiments/reflection_ladder.py       # naming vs listing: the 2,222x
python experiments/reflection_cost_model.py   # flat pricing, and the filters going neutral
python experiments/reflection_options.py      # breadth buys the floor, not the height
python experiments/reflection_retraction.py   # what the interior cannot see

python -m unittest discover -s tests      # 1599 checks, ~15 min
python tools/make_figures.py              # regenerate the figures above
```

Every experiment prints its **pre-registered predictions**, its measurement, and
its verdict — including when the prediction failed. None of them is silent about
losing.

## Layout

```text
project_genesis/     the instruments — field dynamics, capacity, gauge sector,
                     substrates (Hopfield, Kuramoto), and the measures;
                     reflection*.py and *_ladder.py are the formal half
experiments/         one file per question, each self-scoring
tests/               ~1599 checks pinning the central claims so they cannot
                     drift silently
Docs/                the theory, the standalone results, the full record
tools/               figure generation
web/ , viewer/       zero-dependency browser toys sharing the same dynamics
```

Module-by-module detail, and how to run the sandbox, the agents and the browser
toys, is in [`Docs/Usage.md`](Docs/Usage.md).

---

## How the work is done

One loop, repeated: **state a claim → build an instrument → run it → report the
verdict, caveats included.** Four habits keep it honest, all of them learned by
getting something wrong first:

- **Pre-register the prediction and its falsifier**, in the experiment file,
  before running it. Failed predictions stay recorded as failed.
- **Sweep the conventions.** Nearly every confident reading in this repo turned
  out to rest on a constant nobody derived — a threshold, a fitting window, a
  search ceiling. Sweeping them is how you find out which.
- **Quote the headroom, not just the pass.** "It didn't move" is not a result
  until you show the sweep *could* have moved it.
- **Report ranges when the statistics only support ranges.** A number quoted more
  precisely than the thing behind it is the most common error here.
- **Ask what a guard reads, not what it is called.** The most productive single
  habit here, and the last one learned. Perturb one thing a predicate should be
  blind to and see whether it notices. Applied to the formal half it found two
  of six walls mislabelled; applied to the field half it found a texture guard
  that reads `3.0` on structure and `3.0` on pure noise.

Corrections are kept in place rather than tidied away. A programme that never
records having been wrong gives you no way to check the parts that are right.

## Scope

This is a **map and an instrument**, not evidence about the physical world. The
simulations are small, the lattices are finite, and where a result is a
lattice-units signature rather than a physical measurement it says so. Several
of the theory's most interesting claims are marked `[framework]` — coherent
consequences that have not been tested here, and are not asserted.

The foundational theory document is
`Docs/The Universal Recursion Principle (URP) _260312_170343.txt`.
