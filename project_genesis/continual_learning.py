"""Continual learning under the capacity law: κ-as-soil for weights.

The third substrate — and the first where the capacity law meets a domain
with *external* ground truth.  The sector field measured "rooting consumes
soil": stored structure can only take hold where local capacity suffices,
rooting consumes it, and the recovery rate is the persistence↔plasticity
dial (`n3_memory_competition`: a sharp crossover at r⋆, write-once memory
below it, plastic memory above).  The criticality transplant showed the same
law transfers off the lattice.  Continual learning is where those results
become *predictions about learning systems*: catastrophic forgetting vs
intransigence **is** the persistence↔plasticity dilemma.

The mechanism, transplanted verbatim to weights:

- every parameter ``w_j`` carries its own capacity ``κ_j ∈ [0, κ₀]``,
- gradients are **gated by capacity**: ``Δw_j = −η · κ_j · g_j`` — plasticity
  is a *resource*, not a constant,
- the resource obeys the engine's law, consumed by the parameter's own
  update activity and regenerating with slack:

      dκ_j/dt = r·(κ₀ − κ_j) − c·load_j·κ_j ,    load_j = |g_j| / ⟨|g|⟩

  (load is the parameter's share of the gradient, so a parameter that task A
  trained hard has drained soil — task B cannot overwrite it until the soil
  recovers at rate ``r``).  The ODE is linear per step and integrated in
  closed form.

Relation to prior art, stated plainly: this is the same *shape* as synaptic
consolidation regularisers (EWC, Synaptic Intelligence) — per-parameter
protection proportional to past use — but with two URP-specific commitments:
the protection is a **dynamical, regenerating resource** rather than a static
penalty, and its recovery rate is predicted (from the field and Hopfield
measurements) to be a **sharp persistence↔plasticity dial**, not a smooth
regulariser weight.  Whether that buys anything beyond a tuned constant
learning rate is exactly what the experiment must measure, not assume.

The model is a deliberately small numpy MLP (no new dependencies), and the
tasks are permuted-feature classification problems from a fixed random
teacher — the standard interference structure of continual-learning
benchmarks, in miniature.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "make_teacher",
    "make_task",
    "permute_task",
    "make_split_pair",
    "make_concept_tasks",
    "MLP",
    "MultiHeadMLP",
    "KappaSGD",
    "ConstructiveKappaSGD",
    "FunctionalKappaSGD",
    "train_task",
    "accuracy",
]


# ---------------------------------------------------------------------------
# Tasks: permuted-feature classification from a random teacher
# ---------------------------------------------------------------------------

def make_teacher(rng: np.random.Generator, dim: int, hidden: int = 16):
    """A fixed random teacher network defining the base labelling rule."""
    return {"w1": rng.standard_normal((dim, hidden)) / np.sqrt(dim),
            "w2": rng.standard_normal((hidden, 1)) / np.sqrt(hidden)}


def make_task(rng: np.random.Generator, teacher: dict, n: int,
              dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``(X, y)`` with labels from the teacher (balanced-ish 2-class)."""
    x = rng.standard_normal((n, dim))
    score = np.tanh(x @ teacher["w1"]) @ teacher["w2"]
    y = (score[:, 0] > np.median(score)).astype(int)
    return x, y


def permute_task(x: np.ndarray, y: np.ndarray,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A new task from an old one: permute the input features (classic CL).

    **Dense interference**: every parameter is implicated in both tasks, so
    per-parameter capacity drains uniformly — the negative-control condition
    in which capacity gating is predicted to degenerate to global LR decay.
    """
    perm = rng.permutation(x.shape[1])
    return x[:, perm], y.copy()


def make_split_pair(rng: np.random.Generator, dim: int, n: int,
                    off_amplitude: float = 0.3):
    """Two tasks with **structured interference**: disjoint informative halves.

    Task A's labels depend only on the first ``dim//2`` features (the second
    half is low-amplitude noise); task B's only on the second half.  The
    model is shared, but the *load* is heterogeneous across parameters — the
    condition under which selective, regenerating protection can do what a
    global learning rate cannot.  Returns ``((xa, ya), (xb, yb))``.
    """
    half = dim // 2
    t_a = make_teacher(rng, half)
    t_b = make_teacher(rng, half)
    xa_sig, ya = make_task(rng, t_a, n, half)
    xa = np.concatenate(
        [xa_sig, off_amplitude * rng.standard_normal((n, dim - half))], axis=1)
    xb_sig, yb = make_task(rng, t_b, n, half)
    xb = np.concatenate(
        [off_amplitude * rng.standard_normal((n, dim - half)), xb_sig], axis=1)
    return (xa, ya), (xb, yb)


def make_concept_tasks(rng: np.random.Generator, dim: int, n: int):
    """Three tasks with genuine compositional structure, for curriculum runs.

    Concept A is a teacher on the first ``dim//2`` features, concept B a
    teacher on the second half, and the composite task C is **XOR(A, B)** —
    learnable only by representing *both* concepts.  All three tasks share
    the same inputs ``x`` (full-dimensional, unit variance); only the
    labelling differs, so curriculum order is the only thing an experiment
    varies.  Returns ``(x, y_a, y_b, y_c)``.
    """
    half = dim // 2
    t_a = make_teacher(rng, half)
    t_b = make_teacher(rng, half)
    x = rng.standard_normal((n, dim))
    score_a = np.tanh(x[:, :half] @ t_a["w1"]) @ t_a["w2"]
    score_b = np.tanh(x[:, half:] @ t_b["w1"]) @ t_b["w2"]
    y_a = (score_a[:, 0] > np.median(score_a)).astype(int)
    y_b = (score_b[:, 0] > np.median(score_b)).astype(int)
    y_c = (y_a ^ y_b).astype(int)
    return x, y_a, y_b, y_c


# ---------------------------------------------------------------------------
# Model: a small numpy MLP (input → tanh hidden → 2-way softmax)
# ---------------------------------------------------------------------------

class MLP:
    """Two-layer classifier; parameters exposed as a dict of arrays."""

    def __init__(self, dim: int, hidden: int, rng: np.random.Generator):
        self.params = {
            "w1": rng.standard_normal((dim, hidden)) / np.sqrt(dim),
            "b1": np.zeros(hidden),
            "w2": rng.standard_normal((hidden, 2)) / np.sqrt(hidden),
            "b2": np.zeros(2),
        }

    def _forward(self, x: np.ndarray):
        h_pre = x @ self.params["w1"] + self.params["b1"]
        h = np.tanh(h_pre)
        logits = h @ self.params["w2"] + self.params["b2"]
        z = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        return h, p

    def predict(self, x: np.ndarray) -> np.ndarray:
        _, p = self._forward(x)
        return p.argmax(axis=1)

    def loss_and_grads(self, x: np.ndarray, y: np.ndarray):
        """Mean cross-entropy and gradients for every parameter."""
        n = x.shape[0]
        h, p = self._forward(x)
        loss = float(-np.log(p[np.arange(n), y] + 1e-12).mean())
        dlogits = p.copy()
        dlogits[np.arange(n), y] -= 1.0
        dlogits /= n
        grads = {
            "w2": h.T @ dlogits,
            "b2": dlogits.sum(axis=0),
        }
        dh = (dlogits @ self.params["w2"].T) * (1.0 - h ** 2)
        grads["w1"] = x.T @ dh
        grads["b1"] = dh.sum(axis=0)
        return loss, grads


def accuracy(model: MLP, x: np.ndarray, y: np.ndarray,
             head: str | None = None) -> float:
    pred = model.predict(x) if head is None else model.predict(x, head=head)
    return float((pred == y).mean())


class MultiHeadMLP(MLP):
    """Task-incremental variant: one shared hidden layer, one head per task.

    The standard continual-learning protocol for task sequences with
    different labelings — a single shared 2-way readout otherwise guarantees
    near-total interference at the head, swamping any representation-level
    effect an experiment wants to measure.  Heads are named at construction;
    ``loss_and_grads`` / ``predict`` take the active head.
    """

    def __init__(self, dim: int, hidden: int, rng: np.random.Generator,
                 heads: tuple = ("A", "B", "C")):
        self.params = {
            "w1": rng.standard_normal((dim, hidden)) / np.sqrt(dim),
            "b1": np.zeros(hidden),
        }
        for h in heads:
            self.params[f"w2_{h}"] = (rng.standard_normal((hidden, 2))
                                      / np.sqrt(hidden))
            self.params[f"b2_{h}"] = np.zeros(2)

    def _forward(self, x: np.ndarray, head: str):
        h_pre = x @ self.params["w1"] + self.params["b1"]
        h = np.tanh(h_pre)
        logits = h @ self.params[f"w2_{head}"] + self.params[f"b2_{head}"]
        z = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        return h, p

    def predict(self, x: np.ndarray, head: str = "A") -> np.ndarray:
        _, p = self._forward(x, head)
        return p.argmax(axis=1)

    def loss_and_grads(self, x: np.ndarray, y: np.ndarray, head: str = "A"):
        n = x.shape[0]
        h, p = self._forward(x, head)
        loss = float(-np.log(p[np.arange(n), y] + 1e-12).mean())
        dlogits = p.copy()
        dlogits[np.arange(n), y] -= 1.0
        dlogits /= n
        grads = {f"w2_{head}": h.T @ dlogits, f"b2_{head}": dlogits.sum(axis=0)}
        dh = (dlogits @ self.params[f"w2_{head}"].T) * (1.0 - h ** 2)
        grads["w1"] = x.T @ dh
        grads["b1"] = dh.sum(axis=0)
        return loss, grads


# ---------------------------------------------------------------------------
# The optimizer: capacity-gated SGD (κ-as-soil for weights)
# ---------------------------------------------------------------------------

class KappaSGD:
    """SGD whose per-parameter plasticity is a regenerating resource.

    ``recovery=None`` disables the capacity machinery entirely — the step is
    then exactly plain SGD (the baseline path, bit-for-bit).

    ``recovery=0.0`` is the write-once limit: consumed soil never returns.
    """

    def __init__(self, model: MLP, lr: float,
                 recovery: float | None = None,
                 consumption: float = 1.0,
                 kappa0: float = 1.0):
        self.model = model
        self.lr = lr
        self.recovery = recovery
        self.consumption = consumption
        self.kappa0 = kappa0
        self.kappa = ({k: np.full_like(v, kappa0)
                       for k, v in model.params.items()}
                      if recovery is not None else None)

    def new_task(self) -> None:
        """Task-boundary hook (no-op here; see ConstructiveKappaSGD)."""

    def remember(self, x: np.ndarray, y: np.ndarray,
                 head: str | None = None) -> None:
        """Memory hook (no-op here; see FunctionalKappaSGD)."""

    def mean_kappa(self) -> float:
        """Mean capacity across all parameters (1.0 when disabled)."""
        if self.kappa is None:
            return 1.0
        total = sum(k.sum() for k in self.kappa.values())
        count = sum(k.size for k in self.kappa.values())
        return float(total / count)

    def step(self, grads: dict) -> None:
        if self.kappa is None:
            for name, g in grads.items():
                self.model.params[name] -= self.lr * g
            return
        # normalise load by the global mean |g| so the consumption dial is
        # comparable across training phases and tasks
        mean_abs = np.mean([np.abs(g).mean() for g in grads.values()])
        mean_abs = max(mean_abs, 1e-12)
        for name, g in grads.items():
            kap = self.kappa[name]
            self.model.params[name] -= self.lr * kap * g
            load = np.abs(g) / mean_abs
            rate = self.recovery + self.consumption * load
            k_ss = self.recovery * self.kappa0 / np.maximum(rate, 1e-12)
            self.kappa[name] = k_ss + (kap - k_ss) * np.exp(-rate)


class ConstructiveKappaSGD(KappaSGD):
    """The constructive-load extension of the capacity law.

    The curriculum experiment measured the base law's blind spot: κ taxes
    *building-upon* exactly like *overwriting*, so strongly-protected
    foundations become too rigid to compose on.  This variant makes the
    distinction the theory was missing:

    - the **learned displacement** ``d_j = w_j − anchor_j`` records what a
      parameter has committed to across *all* learning so far — the anchor is
      the network's **initialization** and is never reset (a v1 that
      re-anchored at task boundaries erased prior commitments and measured
      catastrophically worse than plain SGD: each new task's drift away from
      old solutions read as "constructive" and passed ungated — kept in the
      record as the definition's own falsifier);
    - an update is **destructive** where it pushes against that commitment
      (``g_j·d_j > 0`` — the step ``−ηg`` would *undo* learned displacement)
      and **constructive** otherwise (extending it, or writing on
      uncommitted parameters);
    - **only destructive updates are gated by κ, and only destructive load
      consumes it.**  Protection against undoing; free passage for building.

    On a fresh network ``d = 0`` everywhere, so all learning is constructive
    — the first task trains exactly like plain SGD.
    """

    def __init__(self, model: MLP, lr: float, recovery: float = 0.1,
                 consumption: float = 1.0, kappa0: float = 1.0):
        super().__init__(model, lr, recovery=recovery,
                         consumption=consumption, kappa0=kappa0)
        self.anchors = {k: v.copy() for k, v in model.params.items()}

    def reanchor(self) -> None:
        """Explicitly reset commitments to the current weights.

        Not called at task boundaries (that was the failed v1 — see the
        class docstring); provided for deliberate curriculum resets only.
        """
        self.anchors = {k: v.copy() for k, v in self.model.params.items()}

    def step(self, grads: dict) -> None:
        mean_abs = np.mean([np.abs(g).mean() for g in grads.values()])
        mean_abs = max(mean_abs, 1e-12)
        for name, g in grads.items():
            kap = self.kappa[name]
            d = self.model.params[name] - self.anchors[name]
            destructive = (g * d) > 0.0
            gate = np.where(destructive, kap, 1.0)
            self.model.params[name] -= self.lr * gate * g
            load = np.where(destructive, np.abs(g), 0.0) / mean_abs
            rate = self.recovery + self.consumption * load
            k_ss = self.recovery * self.kappa0 / np.maximum(rate, 1e-12)
            self.kappa[name] = k_ss + (kap - k_ss) * np.exp(-rate)


class FunctionalKappaSGD(KappaSGD):
    """Function-space constructive κ — the formulation the parametric failure named.

    The per-parameter displacement variants (`ConstructiveKappaSGD`) failed
    because a prior task's solution is an *optimum*: movement in either
    direction destroys it, so no sign test on single weights can tell
    building from breaking.  The distinction is **functional**: destructive
    load is *what degrades prior function*.

    This variant measures it directly.  A small buffer of exemplars is
    stored per completed task (:meth:`remember`, called at task boundaries).
    Each step computes the prior-task gradient ``g_prior`` on the buffer;
    when the step conflicts with prior function (``g·g_prior < 0`` — to
    first order the update raises prior-task loss), the gradient is split
    into its **damaging component** (the projection onto ``g_prior``) and
    its **orthogonal remainder**.  The remainder — new learning the prior
    tasks are flat in — passes fully free; the damaging component is gated
    by per-parameter κ and is what consumes capacity.  (The conflict signal
    and the projection are A-GEM's; the *response* is the capacity law's —
    A-GEM deletes the damaging component outright, here its passage is a
    regenerating budget: fresh capacity permits some interference,
    sustained interference throttles itself, recovery restores plasticity.
    Two failed intermediates are part of this record: an elementwise-sign
    variant read same-task gradient noise as ~50% conflict and drained κ
    at the optimum; a global scalar gate protected too weakly because it
    slowed the whole step instead of the damaging direction.)

    With no memories stored the step is exactly plain SGD, so the first
    task trains untouched.  Honest note: this optimizer uses information
    the others do not (stored exemplars); rehearsal-family baselines are
    the fair external comparison, named in the experiment.
    """

    def __init__(self, model: MLP, lr: float, recovery: float = 0.1,
                 consumption: float = 1.0, kappa0: float = 1.0,
                 buffer_size: int = 64):
        super().__init__(model, lr, recovery=recovery,
                         consumption=consumption, kappa0=kappa0)
        self.buffer_size = buffer_size
        self._memories: list[tuple[np.ndarray, np.ndarray, str | None]] = []

    def remember(self, x: np.ndarray, y: np.ndarray,
                 head: str | None = None) -> None:
        """Store exemplars of a completed task — and consolidate.

        Integration consumes the capacity to *un*-integrate: consolidation
        sets the destructive budget to the law's own steady state under full
        conflicting load, ``κ ← r/(r + c)`` (no new parameter), so protection
        is in place from the first conflicting step rather than arriving
        after the damage (the measured failure of the full-budget variant).
        Recovery ``r`` then gradually returns plasticity — the
        persistence↔plasticity dial, in function space — while sustained
        conflict re-consumes the budget and keeps defended memories pinned.
        """
        n = min(self.buffer_size, x.shape[0])
        self._memories.append((x[:n].copy(), y[:n].copy(), head))
        k_consolidated = self.recovery * self.kappa0 / (self.recovery
                                                        + self.consumption)
        for name in self.kappa:
            np.minimum(self.kappa[name], k_consolidated, out=self.kappa[name])

    def _prior_gradient(self) -> dict:
        """Summed prior-task gradients on the buffers at the current weights."""
        total: dict = {}
        for x, y, head in self._memories:
            if head is None:
                _, grads = self.model.loss_and_grads(x, y)
            else:
                _, grads = self.model.loss_and_grads(x, y, head=head)
            for name, g in grads.items():
                total[name] = total.get(name, 0.0) + g
        return total

    def step(self, grads: dict) -> None:
        if not self._memories:
            for name, g in grads.items():
                self.model.params[name] -= self.lr * g
            return
        g_prior = self._prior_gradient()
        dot, norm_p = 0.0, 0.0
        for name, g in grads.items():
            gp = g_prior.get(name)
            if gp is not None:
                dot += float((g * gp).sum())
                norm_p += float((gp ** 2).sum())
        if dot >= 0.0 or norm_p <= 1e-24:
            # no first-order damage to prior function: fully constructive —
            # the step passes free and the budget recovers (zero load)
            for name, g in grads.items():
                self.model.params[name] -= self.lr * g
                kap = self.kappa[name]
                self.kappa[name] = self.kappa0 + (kap - self.kappa0) * np.exp(
                    -self.recovery)
            return
        coef = dot / norm_p                    # < 0: projection onto g_prior
        mean_abs = np.mean([np.abs(g).mean() for g in grads.values()])
        mean_abs = max(mean_abs, 1e-12)
        for name, g in grads.items():
            kap = self.kappa[name]
            gp = g_prior.get(name)
            g_par = coef * gp if gp is not None else 0.0   # damaging component
            g_orth = g - g_par                             # free construction
            self.model.params[name] -= self.lr * (g_orth + kap * g_par)
            load = np.abs(g_par) / mean_abs
            rate = self.recovery + self.consumption * load
            k_ss = self.recovery * self.kappa0 / np.maximum(rate, 1e-12)
            self.kappa[name] = k_ss + (kap - k_ss) * np.exp(-rate)


def train_task(model: MLP, opt: KappaSGD, x: np.ndarray, y: np.ndarray,
               epochs: int, batch: int, rng: np.random.Generator,
               head: str | None = None) -> None:
    """Mini-batch training of one task (pass ``head`` for MultiHeadMLP)."""
    n = x.shape[0]
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch):
            idx = order[start:start + batch]
            if head is None:
                _, grads = model.loss_and_grads(x[idx], y[idx])
            else:
                _, grads = model.loss_and_grads(x[idx], y[idx], head=head)
            opt.step(grads)
