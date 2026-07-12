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
    "MLP",
    "KappaSGD",
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


def accuracy(model: MLP, x: np.ndarray, y: np.ndarray) -> float:
    return float((model.predict(x) == y).mean())


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


def train_task(model: MLP, opt: KappaSGD, x: np.ndarray, y: np.ndarray,
               epochs: int, batch: int, rng: np.random.Generator) -> None:
    """Mini-batch training of one task."""
    n = x.shape[0]
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch):
            idx = order[start:start + batch]
            _, grads = model.loss_and_grads(x[idx], y[idx])
            opt.step(grads)
