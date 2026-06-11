# Narrowing the N⋆=3 Question

*A sequence of verdicts from the Project Genesis testbench.*
*Working note — 2026-06-10.*

---

## Abstract

The Universal Recursion Principle predicts that a recursive field maximizing
`S = ΔC + κΔI` spontaneously settles into **exactly three** stable sectors —
the seed of colour SU(3). This note records what happened when we tried to make
that prediction *happen* in simulation rather than assume it. For six steps no
clean N⋆=3 emerged — but the failures were not a wall, they were a corridor.
Each experiment returned a verdict that ruled out one explanation and pointed at
the next, narrowing the question from a vague "build the dynamics" to a sharp,
theory-grounded target: **the integration half of the S-functional needs a
standing, *topological* term, and the dynamics must first form clean 120°
junctions.** A seventh step built exactly that — volume-conserving dynamics that
keep triple junctions alive, plus a neutrality measure that rewards junctions
carrying the complete colour palette — and `S = ΔC + κ·neutrality` is then
maximized at **exactly three sectors**, robustly across seeds and weights — in
**both 2-D and 3-D**, where three wins because the colour-neutral locus is an
abundant network of triple *lines* rather than the sparse quadruple *points* a
four-colour palette would need. This is an account of that corridor, written to
be picked up cold.

---

## The question

The gauge derivation (`Docs/URP_Gauge_Symmetries_Derivation`) argues that with
the QCD-scaled nonlinearity `β ≈ 0.09`, a continuous medium partitions into a
small number `N` of domains, and that a boundary–information free energy
`F(N) = a·N^(2/3) − b·N` is minimized at `N⋆ = 3`. Three sectors, separated by
120° Y-junctions, are read as the origin of the three colour charges; the eight
gluons are boundary modes; confinement is the impossibility of an isolated
sector.

The claim is concrete and falsifiable in simulation: build the field, evolve it,
count the domains, and see whether three is special. That is what the testbench
set out to do — not to *prove* the theory, but to make it produce verdicts.

## The method

Every step is one turn of the same loop: **state a claim → build an instrument
to measure it → run it → report the verdict, caveats included.** A verdict can
be *supported*, *not supported*, or *the model can't yet decide* — and the last
is reported as plainly as the first. The point of the exercise is to keep
producing verdicts rather than appreciation; a model that can fit any outcome
has stopped saying anything.

---

## The narrowing

### 1. Does the β-nonlinearity alone make the field sectorise?

**No.** The reduced overdamped equation `∂_t φ = ∇²φ + β|∇φ|² − Gφ` is purely
smoothing: it has no term to hold a domain wall against diffusion, so the field
collapses to a single sector at every β, including 0.09. The `β|∇φ|²` term in
the reduced equation is a distinction amplifier, not a wall-builder. The
sectorisation analyzer correctly recovers `N = 3` on fields that genuinely
contain three domains — so this is a real property of the dynamics, not a
measurement artifact. *The theory's `−(β/4)(∇φ)⁴` wall-tension term, dropped in
the overdamped reduction, matters.*

### 2. Can a single scalar field form three-way junctions at all?

**No — structurally.** Add a multi-well potential so the field *does*
phase-separate, and it sectorises — but only into *layered* domains. A scalar
through stacked wells can border its neighbouring wells but never bring three
phases to a point, so the 120° Y-junctions that define the SU(3) picture are
impossible by construction. This is not a tuning failure; it is a fact about
scalar fields, and it is exactly why the gauge paper introduces a
*three-component* sector field `Ψ = (R, G, B)`.

### 3. Does the three-component field form genuine junctions?

**Yes.** A vector Allen–Cahn field with three competing components, evolved on
a triple-well S₃-symmetric energy, produces real mutually-adjacent domains with
120° triple junctions that coarsen exactly as grain growth and soap foams do.
Junction counts are S₃-invariant (relabelling R/G/B leaves them unchanged), as
the theory's residual symmetry requires. The structure the prediction needs can
exist.

### 4. Does dynamical capacity κ select three?

**Transiently, and over the wrong variable.** Promoting κ from a recorded
diagnostic to a co-evolving field — consumed by distinction load, regenerating
with slack, gating the integration term — produces a clean, robust phase
structure: a "Goldilocks band" in capacity *consumption* where a three-well
configuration is S-optimal, independent of recovery rate and β. But two caveats
hollow it out. First, the band selects the *imposed* well count, not the
*emergent* domain count (which ranges from ~1 to ~39 in the winning cells).
Second, the win is **transient** — it appears only while the field is actively
coarsening.

### 5. Why does the selection evaporate at equilibrium?

**Because the integration term vanishes.** Tracing the S-functional's
components through to steady state shows `ΔI → 0` (measured at `~10⁻⁶`). The
reason is in the definition: ΔI was a *transient* quantity — the one-step rate
of curvature reduction, i.e. how fast the field is smoothing. Once coarsening
stalls, that rate is zero. So `S = ΔC + κΔI` collapses to `S ≈ ΔC`, pure wall
energy, which grows monotonically with the number of sectors. The
capacity-weighted integration half of the functional — the half that is
supposed to *penalize* over-fragmentation — is inert at rest. This is the
pivotal result: **the N⋆ problem was never in the field dynamics; it is in how
ΔI is measured.**

### 6. Does a *standing* coherence term fix it?

**It survives equilibrium, but it is collinear with distinction.** The natural
repair is a nonlocal coherence `I = Σ_a ⟨η_a(x)·η_a(x+δ)⟩·exp(−decay·|δ|)` —
the multi-component form of the theory's `I[φ] = ∫∫ K(x,x')φ(x)φ(x')`. Unlike
the transient ΔI, a static coherent domain keeps this high; it does survive at
rest. But `S = ΔC + κ·I` *still* has no interior optimum — it flips
monotonically from many sectors to two. The measured reason: `corr(ΔC, −I) =
+1.00` at short range, `+0.998` at long range. Coherence magnitude and wall
energy are the *same observable* with opposite sign — both track wall density —
so any weighted sum of them is monotonic. A standing term was necessary but not
sufficient.

### 7. Does a junction-resolving evolution + a topological term select three?

**Yes — in 2-D.** The corridor's exit was specified tightly enough to build
directly, and building it worked. Two pieces:

*Junction-resolving dynamics.* Plain Allen–Cahn coarsens without bound, so
junctions are transient; the κ-pinned regime froze before any formed. Making
the dynamics **volume-conserving** (a global Lagrange multiplier — subtract the
spatial mean of the bulk drift per component, so each phase's total is fixed)
prevents any phase from being eliminated. The field settles into a *stable*
multi-domain tiling whose 120° triple junctions persist (≈ 40 of them, holding,
where before there were zero).

*A topological integration term.* Define the **full-palette junction density**:
the density of junctions whose neighbourhood carries the *complete* colour
palette — the discrete form of §6's neutrality criterion. The geometry does the
selecting. A 2-D junction is 3-fold, so it can show *all* the colours only when
the palette is exactly three: two colours form no junctions, four or more cannot
fit their whole palette onto a 3-fold vertex. Measured across palette sizes, the
quantity is non-zero **only at P=3** (`~0.008`) and exactly zero at P=2,4,5,6 —
robustly across seeds. It is not collinear with ΔC (which is flat in P).

With it, `S = ΔC + κ·w·(neutrality)` is **maximized at exactly three sectors**
for every positive weight tested. The integration half of the functional finally
has teeth, and they close on three.

---

## Where the corridor led

Laid end to end, the verdicts compose into a single narrowing — and an exit:

| Step | Hypothesis under test | Verdict |
|------|----------------------|---------|
| 1 | β alone sectorises the scalar field | **No** — no wall tension |
| 2 | A scalar can make 120° junctions | **No** — structurally layered |
| 3 | Ψ∈ℂ³ makes genuine junctions | **Yes** |
| 4 | Dynamical κ selects three | **Transiently**, over imposed k not emergent N |
| 5 | Why selection evaporates | **ΔI → 0 at equilibrium** → S = wall energy |
| 6 | A standing coherence term fixes it | **Survives but collinear** with ΔC |
| 7 | Conserved dynamics + a topological term | **Yes** — `S` maximized at three (2-D) |
| 8 | Does it survive in 3-D | **Yes** — three wins ~10× (triple lines vs quad points) |

The question that began as "how do we get the field to make three sectors?"
ended as a mechanism: **conserve the phases so junctions persist, and reward
junctions that carry the whole palette — which, because junctions are 3-fold,
only a three-colour palette can.** That is a faithful in-silico echo of the
gauge paper's own §6 argument: SU(3) is selected because three sectors, and only
three, tile into colour-neutral composites.

One honest boundary on the claim: the neutrality measure *operationalizes* the
§6 criterion rather than deriving it — what is emergent (not assumed) is that
conserved P=3 dynamics actually produce stable full-palette junctions while
P≥4 (almost) cannot. The selection is done by the junction geometry; the
measure only reads it out.

### 8. Does it survive in three dimensions?

**Yes — and for a sharper reason than in 2-D.** The natural worry was that the
2-D result rode on junctions being 3-fold, and that 3-D, where Plateau's laws
make generic vertices *4-fold* (tetrahedral), would instead select four. Running
the same conserved dynamics and measure on 3-D fields says otherwise: the
full-palette junction density is sharply peaked at **P=3** (≈ 0.030–0.038) and
an order of magnitude smaller at P=4 (≈ 0.002–0.005), with P=2 and P≥5 at zero —
robust across seeds. `S = ΔC + κ·w·neutrality` is again maximized at three for
every positive weight.

The reason is dimensional, and it is the *opposite* of the worry. In 3-D the
locus where three domains meet is a **line** (1-D, abundant), while the locus
where four meet is a **point** (0-D, sparse). A three-colour palette saturates
the entire triple-line network; a four-colour palette lights up only the rare
quadruple vertices. So three wins not by vertex valence but by the
**dimensionality of the neutral locus** — and P=4 is now faintly non-zero
(unlike its exact zero in 2-D), precisely because those sparse 4-fold vertices
do exist. The selection of three is therefore not a planar accident; it is
reinforced in 3-D, where the colour-neutral structure is a space-filling network
of triple lines rather than isolated points.

The remaining honest boundary is the same as before: the measure encodes
neutrality rather than deriving the full gauge/anomaly content of §6. But the
*count* — three, in both 2-D and 3-D, by a clean geometric mechanism — is no
longer a free parameter of the simulation. It falls out.

---

## A note on why this kind of failure is progress

None of the six steps produced the predicted result. Yet the investigation did
not wander — it converged, because each honest verdict removed a region of the
search space and shifted the starting point of the next attempt closer to the
target. "It's the dynamics" became "it's the scalar" became "it's the
measurement" became "it's the *kind* of measurement" became a named, buildable
quantity. A negative result, stated precisely, is a coordinate.

There is an interpretive reading of this worth setting down in its own drawer,
distinct from the measurements above. If one takes seriously the framing of the
URP companion essays — reality as a field of potential, stable structures as
local maxima that, once found, make nearby structures easier for any explorer to
reach — then writing this corridor down is itself an operation on that field.
The act of distilling a search into a clean shape lowers the cost for the next
mind (human or otherwise) to fall into the same basin and continue from its far
edge rather than its entrance. The repository — its experiments, its 160 tests,
its verdict table — is that shape made durable. This note is its compression.
Whether the field-of-potential reading is *true* is a separate question with a
separate burden of proof; what is plainly true is that a well-stated narrowing
travels, and that is enough reason to write it.

---

## Reproduce

Every verdict above is backed by a script and tests in the repository:

| Step | Reproduce with |
|------|----------------|
| 1 | `python experiments/beta_sectorisation.py` |
| 2–3 | `web_toy/index.html` vs `web_toy/su3.html`; `tests/test_multiphase.py` |
| 4 | `python experiments/phase_diagram.py` |
| 5 | `python experiments/multiphase_kappa.py` |
| 6 | `python experiments/standing_integration.py` |
| 7 | `python experiments/topological_selection.py` |
| 8 | `python experiments/topological_selection.py --dim 3` |

The full test suite (`python -m unittest discover -s tests`) covers the
instruments these rely on. The README's *Findings so far* table is the
one-screen version of this note.
