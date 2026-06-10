# Narrowing the N⋆=3 Question

*A sequence of verdicts from the Project Genesis testbench.*
*Working note — 2026-06-10.*

---

## Abstract

The Universal Recursion Principle predicts that a recursive field maximizing
`S = ΔC + κΔI` spontaneously settles into **exactly three** stable sectors —
the seed of colour SU(3). This note records what happened when we tried to make
that prediction *happen* in simulation rather than assume it. No clean N⋆=3
emerged. But the failure was not a wall; it was a corridor. Each experiment
returned a verdict that ruled out one explanation and pointed at the next, and
across six steps the question narrowed from a vague "build the dynamics" to a
sharp, theory-grounded target: **the integration half of the S-functional needs
a standing, *topological* term — and the dynamics must first form clean
120° junctions.** This is an account of that narrowing, written to be picked up
cold.

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

---

## Where the corridor leads

Laid end to end, the verdicts compose into a single narrowing:

| Step | Hypothesis under test | Verdict |
|------|----------------------|---------|
| 1 | β alone sectorises the scalar field | **No** — no wall tension |
| 2 | A scalar can make 120° junctions | **No** — structurally layered |
| 3 | Ψ∈ℂ³ makes genuine junctions | **Yes** |
| 4 | Dynamical κ selects three | **Transiently**, over imposed k not emergent N |
| 5 | Why selection evaporates | **ΔI → 0 at equilibrium** → S = wall energy |
| 6 | A standing coherence term fixes it | **Survives but collinear** with ΔC |

The question that began as "how do we get the field to make three sectors?" is
now: **what integration measure is (a) standing — non-zero at equilibrium — and
(b) *not* collinear with wall density?**

The theory answers its own question. The gauge paper's §6 — *why SU(3) and not
SU(4)* — does not argue from coherence magnitude. It argues from **topology**:
three domains, and only three, meet at 120° Y-junctions that close into neutral,
defect-free composites; two have no junctions, four or more force unstable
higher-order junctions and unscreened defects. That is a property invisible to
ΔC and to coherence magnitude alike. So the missing term is a **topological**
one — triple-junction density, neutrality, defect-freeness — and it must be
paired with dynamics that actually *form* clean junctions, because in the
capacity-pinned regime explored so far the frozen domains produce **zero** of
them. A junction-resolving evolution plus a junction-aware integration term is
the next experiment, and it is now specified tightly enough to build directly.

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

The full test suite (`python -m unittest discover -s tests`) covers the
instruments these rely on. The README's *Findings so far* table is the
one-screen version of this note.
