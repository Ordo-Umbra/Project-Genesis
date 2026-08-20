# The Reflection Ladder

### What thirteen experiments actually showed, in plain language

*A companion to [`The_Generative_Gap.md`](The_Generative_Gap.md), written to be
read cold. No proof theory assumed. Every claim here is either something a
program measured, something cited from the literature, or something explicitly
flagged as neither — and those three are kept apart on purpose, because the
whole value of the exercise is knowing which is which.*

---

## The short version

You can build a tower of mathematical systems where each floor proves something
the floor below it couldn't. The tower never tops out. That much has been known
since Gödel and it is not what we measured.

What we measured is that **the tower's behaviour depends almost entirely on how
each floor writes down its own address** — and that the bookkeeping people
usually use to describe such towers cannot see the difference. Two towers with
identical descriptions on paper can differ by a factor of 2,222 in cost, or
differ in whether they are doing anything at all, or differ in whether they can
go on past a certain height *at any price*.

And at the top, a system can always work out **where** it will stop, but not
always **whether it had to**.

---

## 1. The object: a tower of theories

Start with ordinary arithmetic — the rules for counting, adding and multiplying
whole numbers. Call it `PA`.

Gödel's second incompleteness theorem says: `PA` cannot prove its own
consistency. It cannot establish, from inside, that its rules never lead to a
contradiction.

So build a new system: take `PA` and add, as a new rule, the statement *"PA is
consistent."* Call it `T₁`. It is strictly stronger — it proves something `PA`
could not.

But `T₁` now can't prove *its own* consistency. So build `T₂ = T₁ + "T₁ is
consistent."` And so on:

```
T₀ = PA
T₁ = T₀ + Con(T₀)
T₂ = T₁ + Con(T₁)
...
```

This is the **reflection ladder**. Each rung genuinely adds something. It never
finishes. `The_Generative_Gap.md` §3 had been citing this ladder for a long time
as the formal twin of the field program's capacity gap. It had never been run.

---

## 2. The trap: two of the three quantities are definitions

The framework describes such a system with three quantities:

| symbol | name | meaning |
|---|---|---|
| `C` | capacity | the ceiling — how high the tower could go in principle |
| `I` | integration | how high it currently is |
| `G` | generative headroom | whether there's another floor available |

with the condition `G > 0` — *there is always another floor* — as the claim
that the system never terminates.

**Here is the problem.** In this setting `C` is a constant of the subject
matter, fixed before anything runs. And `G > 0` holds because we *defined* the
next floor to exist — `T_{n+1} = T_n + Con(T_n)` is a construction, not a
discovery. So a program that ran the ladder and reported "`C` is still the
ceiling, `I` went up by one, `G` is still positive" would be a tautology with a
progress bar. Nothing it printed could have come out otherwise.

The same trap sits in the obvious measurement. "Each floor proves the
consistency of the one below" is true because we *put that sentence in*. And
"the floor below can't prove it" is Gödel's theorem — cited, not observed.

**So every experiment here is built around a quantity that can come out other
than the model says, and each has a deliberately broken control arm that makes
it do so.** That is the entire methodology and everything below depends on it.

The measured quantity is: **did this rung actually enlarge the set of rules?**
One bit per floor. Plus a machine-checked proof that the added rule can be
*used*, not merely stored.

---

## 3. Result one — naming versus listing

### The intuition

An idea that is well-defined and compact stays with you and costs almost nothing
to carry. An idea you can only convey by reciting every detail is expensive
every single time.

### What the math actually says

When `T_{n+1}` says *"`T_n` is consistent"*, it has to refer to `T_n` somehow. It
needs an **address** for `T_n`'s rulebook. There are infinitely many valid
addresses, and — this is the load-bearing fact — **the mathematics does not
specify which one to use.** It is a free choice.

Two honest choices:

- **`inline`** — the address *is the entire rulebook, written out.* To say
  "`T_n` is consistent", floor `n+1` must carry a complete transcript of
  everything below it.
- **`indexed`** — the address is a *description*: "arithmetic, plus the first
  `n` floors of this tower." It never writes out what it refers to.

These build genuinely different towers (different addresses make literally
different sentences), but the *same construction* under two naming schemes.

### What was measured

At floor 12:

| scheme | size of the presentation | growth per floor |
|---|---|---|
| `inline` | **11,103,120 symbols** | ×2.00 — doubles |
| `indexed` | **4,996 symbols** | ×1.00 — flat |

A factor of **2,222**, on two towers whose floors are *one-for-one identical in
what they add*. Every floor of both was productive, 12 out of 12, and the
conjunction of all twelve added rules was derived and machine-checked in 45
lines — each one used as a premise, not just filed.

Then a third, deliberately broken arm — **`truncated`** — whose address has a
fixed-width counter, like a 3-digit odometer. After 8 floors the counter wraps.
The address it produces is one it has used before. The sentence it adds is one
it already has. **The tower stops growing while the floor counter keeps
climbing** — 16 rules, frozen, at floor 8, 9, 10, 11.

### What this shows

The dissociation runs in **both** directions:

- **Size is not evidence of rank.** `inline` vs `indexed`: identical progress,
  2,222× different cost. You can inflate the paperwork without limit.
- **Rank is not evidence of capability.** `indexed` vs `truncated`: identical
  cost, and one is working while the other has stopped. The counter says nothing.

So the natural question — *does climbing higher buy real capability, or just
formal complexity?* — has **neither** answer. It buys capability only when the
naming scheme lets the step be productive, and whether it does is a separate
fact that has to be measured.

### What it does not show

Nothing about *which* naming scheme a real system would use. `inline` is a
strawman in the sense that no one would design it — but it is the scheme you get
by default if you think of a system as "the list of things it contains" rather
than "the description of what it contains", and that is a common way to think.

---

## 4. Result two — the missing budget

### The intuition

Anything real has to be paid for. A process that continues forever *for free*
isn't a model of anything.

### What the math actually says

The field program's whole content is that integration is expensive and capacity
is the budget it comes out of — `∂ₜκ = r(κ₀−κ) − load`, with a recovery rate `r`
that decides whether the system can keep paying.

The tower had **no budget at all.** Reflection was free. Which means "the tower
never stops" was a statement about how we defined the available moves, not about
the tower. `The_Generative_Gap.md`'s own comparison table maps "capacity pays
for integration" onto "proof strength buys reach" — and nothing on the
proof-theory side ever ran out.

So we gave it one: **a floor is reachable only if the tower can afford to build
it**, paid from a capacity that heals at rate `r` between floors.

### What was measured

**Endings became possible.** The doubling arm dies at every budget:

| budget | 10⁴ | 10⁵ | 10⁶ | 10⁷ | 10⁸ | 10⁹ | 10¹⁰ |
|---|---|---|---|---|---|---|---|
| floor reached | 2 | 6 | 9 | 12 | 16 | 19 | 22 |

**Capacity buys almost nothing against doubling cost:** reach grows at exactly
**1.000 floors per doubling of budget**. Six orders of magnitude of extra
capacity bought 20 floors. A scheme whose costs double cannot be rescued by
money — only by a better scheme. (This is the tower's version of the field
program's *the separation cannot be bought*.)

**The recovery rate is a sharp dial, at a value we derived first.** Paying `L`
and healing a fraction `r` back toward the ceiling settles at
`κ* = κ_max − L(1−r)/r`, which is sustainable exactly when `κ* ≥ L`, giving a
critical rate `r* = L/κ_max`. Measured by bisection across three budgets and both
flat-cost arms, it matched to **better than 0.01%**. Below `r*` the budget drifts
down and the tower dies; above it, the budget settles at a level that pays
forever.

**But a budget alone can't tell a real tower from a broken one.** The
`truncated` arm takes exactly the same number of floors as `indexed`, with an
identical critical recovery rate — indistinguishable on every cost
measurement — while producing 8 new rules to `indexed`'s 64.

### What this shows

Ranking systems by *how long they can keep going* actively rewards the one doing
the least. Cost-bounding rules out steps that are too expensive and says nothing
at all about steps that are cheap and pointless.

**The corrected picture needs both halves:** a move is available only if it is
*affordable* **and** *productive*. Restrict it that way and the broken arm stops
at floor 8 at every budget and every recovery rate, while the working one runs
on. The budget makes `G > 0` capable of being false; the productivity
certificate makes it *mean* something.

### What it does not show

The cost model is a choice (the cost of *building* the next floor, not of
*holding* the tower). A different choice moves the constants, not the ordering.
And the derived `r*` is exact only for flat-cost arms, which is why it is
reported for those and withheld from the doubling one.

---

## 5. Result three — where a price becomes a gate

### The intuition

Some limits you can pay your way past. Some you can't, and no amount of money
changes it.

### What the math actually says

There is a second way to move, besides "add one more floor". You can take the
**limit**: stand on top of *everything built so far, all at once*, and treat that
as a new starting point. `T_ω = ⋃ₙ Tₙ`.

To do that you have to *name* the union of the whole infinite tower below you.

- A naming scheme that writes **descriptions** can do this. "Arithmetic plus
  every floor below this point" is a description exactly as short as any other.
- A naming scheme that writes **lists** cannot. There is no finite list of an
  infinite union. Not an expensive one — **there isn't one.**

### What was measured

Watch what happens to the doubling arm as its budget rises:

| budget | stops at | why |
|---|---|---|
| 10⁵ | floor 6 | can't afford it |
| 10⁶ | floor 9 | can't afford it |
| 10⁹ | floor 12 | **no limit exists** |
| 10¹² | floor 12 | no limit exists |
| 10¹⁵ | floor 12 | no limit exists |

**The wall moves while it's about money, and then it stops moving and starts
refusing.** Ten thousand times more budget buys nothing, because the obstacle
changed *kind*.

And where the limit *is* available, it costs **exactly what an ordinary floor
costs** — ratio 1.000, flat regardless of how much tower it subsumes.

### What this shows

What a good naming scheme buys is not a cheaper list. It is **the right to stop
listing.** And that is not a discount — it is a capability that the other scheme
does not have at any price.

There is a lovely detail in the broken arm here too: taking a limit **restarts**
it. A tower frozen by its wrapped counter starts producing again after a limit —
then freezes again 8 floors later, forever. `[9, 8, 8, 8, 8]` per block. So a
limit rescues a dead tower, but the reprieve is bounded by the same defect that
killed it. It buys blocks, not a rate.

---

## 6. Result four — going up levels is free; what Kleene's `O` really costs

### The intuition

If you can take a limit, you can take a limit *of limits*, and keep going up
levels. Surely at some point this gets harder.

### What the math actually says

It doesn't. Ordinals below `ω^ω` are written in **Cantor normal form** —
`ω^k·a_k + … + ω·a₁ + a₀` — which is just a list of coefficients. "Take a limit
at level `j`" means: bump coefficient `j`, zero everything below. Going from `ω`
to `ω²` bumps a different coefficient than going from 3 to 4. It's the same move.

### What was measured

Cost ratio to an ordinary floor, at every level from 1 to 6: **1.0000**,
productive at each. Cost per step flat to **0.00%** all the way to `ω⁶+1`,
reached in 30 steps, every one of them productive.

**Rank is not bought with cost at all.** It's bought with the right to name, and
once a scheme has that right, how high it goes is limited only by how many times
you apply it.

### The answer on Kleene's `O`

`O` is the notation system for *all* the recursive ordinals — the full ceiling.
The open question was whether reaching `ω²` forces us to pay for it.

**It doesn't**, and neither does `ω^ω`, nor `ε₀`. Those all have ordinary
notation systems with unique representations and decidable comparison. The
deferral has cost nothing and would keep costing nothing for a very long way up.

What `O` buys is *everything*. What it **spends** is this: below it, a
fundamental sequence is closed-form arithmetic on the normal form — nothing
runs, so nothing can fail to halt. In `O`, a limit address is an arbitrary
program, and the address is valid only if that program is *total* (halts on
every input). That is undecidable in general.

**So the price of `O` is not implementation effort. It is the decidability of
your own list of available moves.**

---

## 7. Result five — complete about location, incomplete about necessity

### The intuition

From the inside, continuing feels unbounded. You never encounter a wall marked
*wall*; you just never find the edge.

### What the math actually says

Every measurement above was taken from **outside**. The observer running the
tower and reading off costs was us, the whole time. So we asked the first
question the tower can put to *itself*: using only checks it can run on its own
description, what does it know about where and why it stops?

This required a fourth arm — **`searched`** — whose limit address is an
arbitrary program rather than a canonical formula, so certifying it means
*running* it. The detail everything rests on: **its program is `n ↦ n`, which is
total. Its next move genuinely exists.**

### What was measured

| arm | budget | predicts | is the wall real? | actually |
|---|---|---|---|---|
| `inline` | 10⁵ | floor 6, unaffordable | yes | floor 6, unaffordable |
| `inline` | 10¹² | floor 10, no limit exists | yes | floor 10, no limit exists |
| `indexed` | any | never stops | yes | never stops |
| `searched` | any | **floor 10** | **cannot tell** | floor 10 |

**16 out of 16 exact on *where*.** Every arm, every budget, including the
undecidable one — because a cautious system knows it will decline a move it
can't certify, so it knows precisely where that happens.

But the `searched` arm **cannot establish whether anything was there.** It halts
on a live edge.

### What this shows

The tempting statement is wrong. It is *not* "from the inside a system can't see
its walls" — it sees all three and knows exactly where each one is. The finding
is narrower and sharper:

> **A system is complete about location and incomplete about necessity.**
> It always knows *where* it stops. It does not always know whether stopping
> was *required*.

At the third wall it cannot distinguish a continuation it **lacks** from one it
merely cannot **certify**.

Which bounds `G` from a new side. `G > 0` is a claim about which moves exist. A
system can always compute where its own climb ends. What it cannot always
compute is whether that ending was `G = 0`, or `G > 0` with no certificate.
**Different facts about the world; identical facts from the inside.**

### And the third wall's position is a choice

The `searched` arm stops because of a **policy** — refuse moves you can't
certify — not because the world stopped it. Lower that requirement and the same
tower continues: measured, it reaches `ω·3+11` with 43 productive steps,
*identical to the working arm*. It was right to continue. It could not have
known that.

The other two walls don't budge under the same change. An absent edge stays
absent; a spent budget stays spent.

So **"when is it over" is, for exactly one of the three walls, a question about
how much certainty you demand rather than about the world.** Demand less and you
go further and are sometimes right; demand less and you may also walk off a
cliff, and nothing available to you distinguishes those two cases in advance.

---

## 8. The three walls

Everything above converges on this table, which is the most compressed statement
of what the series found:

| ending | what it means | kind | moves with budget? | moves with policy? | visible from inside? |
|---|---|---|---|---|---|
| `unaffordable` | the move exists, too expensive | economic | **yes** | no | fully |
| `limit-undefined` | the move does not exist | structural | no | no | fully |
| `undecidable` | can't be determined whether it exists | epistemic | no | **yes** | location only |

The `(C, I, G)` bookkeeping distinguishes none of them. That is the honest
summary of what these five experiments contribute: not that the framework is
wrong, but that it is **under-determined** — it has one moving part where the
measurements found at least four, and the ones it lacks are the ones that decide
whether a system is actually getting anywhere.

---

## 8b. Result six — G, derived rather than posited

*Added after two independent reviews of this document converged on the same
directive: stop treating `G` as a primitive.*

### The intuition

If a quantity can be computed from things you already measure, it should not
also be assumed.

### What the math actually says

`G` was carried as an independent magnitude, with `G > 0` asserted. But the five
results above found **four** separate things that can stop a system, and `G`
could express none of them. So compute it from them instead:

    G_actual    = structural ∧ affordable ∧ productive
    G_certified = certifiable ∧ affordable ∧ productive

The first three are facts about the world; the fourth is a fact about what the
system can know. Three verdicts follow, and the middle one is new:

| verdict | condition | halts? | meaning |
|---|---|---|---|
| `terminal` | no move exists at all | yes | nothing is there |
| `stagnant` | moves exist, none productive | **no** | it keeps going and gets nowhere |
| **`hidden`** | productive move exists, uncertified | yes | **something is there and not known** |
| `recognised` | productive move exists, certified | no | something is there and known |

*Four, not three.* A first version of this table said `terminal` when
`G_actual = 0` and `recognised` when `G_actual = G_certified`, and a review
caught two problems with it. The degenerate case `G_actual = G_certified = 0`
satisfies both. And more seriously, having **no move** and having **only
useless moves** are different situations — the second one does not halt, which
is exactly what §8b's own measurement found and the table then contradicted.

### What was measured

**12 out of 12.** For every arm at every budget, the derived verdict names the
same wall an actual climb reports.

And checked exhaustively over all 16 sound combinations of the four predicates,
**the three walls do not classify terminal states**. Terminal splits only two
ways — economic and structural. The epistemic case is not terminal at all: it
has a live move (`G_actual = 1`) and halts anyway. Halting and being out of
moves are different cuts through the same space, and collapsing them loses
precisely the distinction §7 exists to draw. No fifth dimension was needed. `hidden` is
realised by exactly one arm — `searched` — which is the epistemic wall of §7,
now falling out of the decomposition rather than being stipulated.

**And unproductivity turns out not to be a wall at all.** The stalled arm runs
clean to the horizon: it is not *stopped*, it *arrives having done nothing*. So
productivity is a fourth dimension **orthogonal** to the three walls. A system
can fail by halting, or by running forever without moving, and those are
different failures — the second one leaves every wall-detector reading normal.

**The cost of demanding evidence has a shape.** Sweeping the requirement `N`
over 400 candidate addresses (half genuinely total, the rest diverging on a
heavy tail): the cliff rate falls from 0.500 to 0.407 and never reaches zero,
while verification cost rises linearly. Every total address is accepted at every
`N`, so the trade-off is pure — more evidence buys precision and nothing else,
at a diminishing rate.

**The strict policy is the corner of that curve, not a point on it.** Zero
cliffs, and zero continuations, forever. A system that demands proof gets
perfect safety and goes nowhere. A system that demands `N` steps goes everywhere
and is sometimes wrong. There is no `N` that is both, and no amount of evidence
converts the second kind of system into the first.

### What it does not show

The population is **constructed** — we choose the divergence points, so the
absolute cliff rate means nothing on its own. Only the shape survives.

---

## 8c. Result seven — a ladder in a box

*Added after a reviewer asked whether the taxonomy survives when `C` varies.*

### The intuition

Everything above was measured in a setting with no ceiling. What happens in a
system that can actually fill up?

### What the math actually says

`I < C` is a **theorem** in the arithmetic setting, so one possible blocker could
never occur: running out of domain. And raising `C` doesn't test it — `I < C` is
a theorem in second-order arithmetic and set theory too. So the *domain* has to
change: `k` atoms, a theory is a subset of them, `C = k`, ceiling reachable.

The question is whether **exhausted** is a real category or stagnation renamed.
They look identical from outside — moves exist, none productive, system keeps
running. One test separates them: **is it rescuable by renaming?**

### What was measured

**Exhaustion is naming-invariant.** Every adequate scheme stalls at exactly `C`.
**Stagnation is not** — a scheme with too small an address space stalls early,
and switching schemes moves it. So `exhausted` is a genuine fifth category.

**And a naming defect is invisible when the box binds first.** Truncated and
indexed are indistinguishable *exactly* when the address space is at least the
box size — 15/15 rows, no exceptions. The `truncated` pathology was detectable
earlier **only because arithmetic is infinite**. In a small enough box, a broken
system and a sound one agree on every observable: same stall, same reason, same
interior prediction. The defect wasn't fixed; the box was too small to show it.

**Exhaustion is fully visible from inside** — 72/72 exact, lookahead unbounded,
known at rung 0. The bookend: the epistemic wall was the one limit a system could
never see coming; this is the one it sees most clearly.

### What it does not show

The box is a model, chosen as the simplest thing that saturates. Whether a richer
saturating domain gives the same five categories is untested.

---

## 8d. Result eight — a ceiling nobody chose

*The box was suspiciously tidy, so this is the same question in a domain where
saturation is not hand-set.*

### What the math actually says

In the box, saturation was **stipulated**: `k` atoms, one added per step. The
address and the thing added were the same object — the condition under which
nothing can come apart. So: `n` propositional variables, `2ⁿ` valuations, a
theory *is* its surviving models, a step eliminates one, and it cannot empty the
set because a theory with no models is inconsistent. `C = 2ⁿ − 1` falls out of
the semantics, and the address now names *which* model to remove.

### What was measured

**The ceiling is still naming-invariant in location.** Every adequate scheme
reaches exactly the same floor.

**But not in cost.** One adequate scheme pays **3.5× more** for the identical
result (n = 8: 255 steps vs 897). The box couldn't show this. Separate the
address from the thing added, and "adequate" splits into *efficient* and
*merely-eventually-correct*.

**And content-addressing turns out to be catastrophic here, not just expensive.**
`inline` names itself by its surviving models, so its address only changes when
an elimination *succeeds*. One wasted move and it is deterministically stuck —
`I = 1` out of `C = 255`, forever. A system whose state is its address cannot
recover from a single move that achieves nothing.

**`stagnant` was hiding a family.** Three mechanisms, all reported identically,
stalling at 1, 128 and 8 in the same domain: **freeze** (address stops moving),
**coverage** (addresses reach only half the space), **collision** (address space
too small).

### What it does not show

The 3.5× is a property of one hash — at `n ≤ 6` the same hash is a permutation
and the gap vanishes. Existence is the claim, not a constant.

---

## 8e. Result nine — productive forever, going nowhere

*The first prediction in this series written down **before** the domain was
built, with the domain chosen by someone else. That was the whole point: §8d
admitted that "each richer domain adds a category" was partly selection, since
each domain had been picked because it could express something new.*

### What the math actually says

A consistency statement can be taken relative to **any finite set** of theories,
not just the previous one in a chain. That turns the ladder into a DAG, and
separates two things every previous domain fused: *this step added something
new* and *the collection reached further*.

### What was measured

**The predicted split exists.** Steps that add a genuinely new sentence and leave
the global rank exactly where it was.

**It needs branching, and branching can't arise from one root.** From a single
base, multi-parent reflection produces **nothing** — 30/30 duplicates, because
every join over a chain carries the content of its maximum. Independent starting
theories have to be *given*. Neither registered prediction said that.

**And it is not a sixth wall.** From one and the same graph, one policy gives a
sideways step and another gives an advancing one. Every earlier category is a
fact about *where a system is*; this is a fact about *what it chose*. Nothing
blocks it — every check passes.

**The consequence:** 200 consecutive productive steps, 209 nodes built, distinct
global ranks visited: **`[5]`**. A system can be **productive forever and go
nowhere.** Same budget spent differently: 39 nodes either way, rank 35 vs 5.

### What it does not show

`depth` is a declared proxy for proof-theoretic rank — the prediction was phrased
in terms of genuine ordinal height, and this is a weakening. And the survival of
the three walls is only *partially* tested here: this domain has no budget, no
naming defect and no certification requirement, so they have nothing to bite on.

---

## 8f. Result ten — the walls select for going nowhere

*The reviewer's next registered run: restore the three filters and ask whether
sideways trajectories survive when any of them can bite.*

### What was predicted, before building

Not merely that they'd fail, but that they'd **preferentially block advancing** —
because every filter scales with *what a step reflects on*, and advancing means
reflecting on the frontier, the largest object in the graph.

### What was measured

| filter | advancing | sideways |
|---|---|---|
| budget 5 | **0/30** | 28/30 |
| 3-bit address | 1/30 | 2/30 |
| certify effort 5 | **0/30** | 28/30 |
| **max 1 parent** | 30/30 | **0/30** |

**At no setting of any of the three does the direction reverse.** Advancing
reflects on a key of size 9 where sideways reflects on 2, and the gap widens with
depth. Every filter reading the reflected object bites advancing first.

**One thing does block sideways: an arity cap.** But it charges for **breadth**,
not depth — it is the first constraint in this series that is *not a function of
the reflected object*.

So the walls meant to make `G` meaningful push a system toward the one trajectory
that satisfies every predicate and gets nowhere. Under a budget it's sharper
still: sideways converts an unbounded wander into a **bounded** one, spending the
whole budget at constant rank with every check passing on the way down.

### What it does not show

With `cost = |key|` and certification compared against `|key|`, the economic and
epistemic filters are the **same predicate** — their agreement is arithmetic, not
evidence. Two independent results here, not three.

---

## 8g. Result eleven — a rank-aware rule restores motion, not advance

*The reviewer's next registered question: does a selection term that reads the
rank order rather than object size restore advance under the filters that
suppressed it?*

### What was predicted, before building

It restores advance only temporarily and **self-defeats** — the frontier is the
most expensive move *because* it is the frontier, and every advance grows the
frontier's closure by one, raising the price of the next.

### What was measured

**Rank saturates at exactly the budget.** 6→6, 10→10, 20→20, 50→50.

**And the gain over blind frontier-seeking is zero.**

| budget | blind deepen | rank-aware | gain | blind refused | aware sideways |
|---|---|---|---|---|---|
| 10 | 10 | 10 | **0** | 55 | 55 |
| 50 | 50 | 50 | **0** | 15 | 15 |

Zero at every budget and under all three filters. The rank-aware rule reaches
**exactly** the rank blind deepening reaches. What it changes is bookkeeping:
blind deepening hits the wall and is *refused* for the rest of the run; the
rank-aware rule hits the same wall and converts those refusals **one-for-one**
into sideways moves.

So it restores **motion**, not advance — and that is arguably worse, because a
refusal is visible as a block while a sideways move passes every check.

**A policy cannot outrun this, because the policy does not set the price.** Only
a cost model that does not grow with the reflected object could, and none of the
six domains had one.

### A retraction, and a correction to the retraction

§8f reported that an arity cap "blocks sideways completely". The cap *does* block
all thirty joins that policy proposed — the measurement stands. The error was
generalising *"joins are blocked"* to *"sideways is blocked"*: a sideways move
needs no join, and single-parent reflections below the frontier are admitted.

My first diagnosis of that retraction was itself wrong — I attributed the zero to
duplicates rather than to arity blocks, and a test assertion caught it. Recorded
because it is the same kind of over-reading that produced the claim being
withdrawn.

---

## 8h. Result twelve — the missing piece was §3

*The reviewer named the last structural gap: no domain had a cost model that
doesn't grow with the reflected object. It turned out not to need a new object —
and swapping the cost model exposed a category that had been mislabelled since
the filters were built.*

### What the math actually says

**§3 already measured one.** `inline` prices by *content* — the address is the
axiom list — and doubles every rung. `indexed` prices by *description* —
"arithmetic plus everything below here" — and stays flat at 4,996 symbols per
rung **however much theory it names.**

The later domains had reverted to content-addressed pricing (`|key|`) without
anyone noticing, me included. So: swap in description pricing and re-run.

### What was measured

**Saturation disappears.** Content-addressed, rank is 10 at every step count —
the budget, and nothing more. Description-addressed: 20 → 25, 40 → 45, 80 → 85,
160 → 165. Unbounded, and **independent of the budget entirely.**

**The filters go neutral.** 9/9 rows block advancing under content pricing; 9/9
are neutral under description pricing. *The bias was never a property of the
filters — it was a property of what they were reading.*

**The sideways basin survives.** A join-seeking policy still lands 30/30 sideways
moves. It isn't removed; **it stops being downhill.**

### The chain, with every link now visible

Naming by description → flat cost → neutral filters → no selection pressure
toward sideways → unbounded advance. **Every link had been measured. What was
missing was noticing they were one chain.**

It also reinstates §4: `r* = L/κ_max` assumed *constant cost*, so it was quietly
inapplicable to every domain after it. Under description pricing it applies
again.

### And a wall that turned out not to be a wall

The reviewer's closing sentence contained a clause nobody had checked: that a
rank-following policy advances without bound *until an independent wall
intervenes* — undecidability of an address being one such wall. Checking it
broke something on my side, not theirs.

The DAG domain's "epistemic" filter admits a step when `cost(key) ≤ effort`.
**That is the same number the budget reads.** Under content pricing you can't
see it, because everything reads size and the three filters look like three
things. Under flat pricing the disguise falls off: a 2-element key and a
20-element key are priced identically, so *no* effort level tells them apart —
every bound admits both or refuses both. It was a third size tax with an
epistemic label on it.

So a real one was built: `opaque`, addresses whose validity **cannot be
settled** — §3's `searched` arm, where certification is a *search that cannot
conclude* rather than a fee that grows. The test it has to pass is simple. A
wall that doesn't read size must not care what size costs.

**It doesn't. 3/3 rows identical across both cost models**, every count matching
exactly.

Two things follow that a size tax cannot do:

- **Its direction is set by placement, not by size.** A size tax always blocks
  advancing, because advancing always enlarges the key — that is not a choice it
  makes, it's arithmetic. This wall has no direction of its own. Mark the root
  the deepening chain runs through and everything stops (30/30 refused, both
  policies). Mark a root off that chain and only the joins are refused (0 vs
  29/30). Same wall, opposite effect, decided entirely by *where the
  unsettleable thing sits*.
- **You can walk around it, for a price.** One unsettleable root costs 5 rungs
  out of 35: the rank-following policy abandons that region and rebuilds
  elsewhere. A detour, not a ceiling. **Unless there is nowhere to go** — mark
  every root opaque and the climb stops dead with all 30 attempts refused, which
  is precisely §3's `searched` arm, where the search runs over the whole address
  space.

**The reviewer was right and my model was wrong.** Those are two findings, not
one, and the second is the more useful: a category can be wrong for eleven
experiments and look fine, as long as nothing ever forces its label and its
behaviour apart. The cost-model swap was what forced them apart. This is the
same failure shape as the three earlier bugs — a well-formed, confident answer
to a *different question* — and it was caught the same way: by a second,
independent route to the number.

### What it does not show

The flat rate is a parameter; its value is arbitrary. What matters is that it
doesn't read the key.

`opaque` is a **declared** unsettleable set. Nothing here proves any address
undecidable — it models what a policy does when one is. The citation for genuine
undecidability is still a citation.

---

## 8i. Result thirteen — ask what a wall reads, not what it is called

*The mislabelled filter was an accident. This turns the accident into a tool,
points it at everything, and catches a second one.*

### The intuition

You can't tell what a rule is enforcing by reading its name. You can tell by
**changing one thing and seeing whether the rule notices.**

Two changes, each leaving the structure alone:

- **Change the price.** Swap content-addressed for description-addressed
  pricing. A wall that moves was reading the price.
- **Change the numbering.** Shift every node's identifier by 8 and relabel the
  wall to match. Same graph, same wall, different numbers. A wall that moves was
  reading *the number*, which is an artifact of how things are written down, not
  a fact about them.

### What was measured

| wall | reads price | reads numbering | what it actually is |
|---|---|---|---|
| `budget` | yes | no | a size tax, honestly labelled |
| `certify_effort` | yes | no | a size tax, labelled epistemic |
| `address_bits` | **yes** | **yes** | a size tax **plus an encoding artifact** |
| `opaque` | no | no | reads neither |
| `opaque_form` | no | no | reads neither |
| `max_arity` | no | no | reads neither |

**The instrument caught one nobody had complained about.** `address_bits` — the
*structural* wall, the one that decides whether an address can be written down
at all — compares the largest identifier in the key against `2^bits`. So
`{0,1,2}` fits in 3 bits and `{8,9,10}` does not, *for the same graph shape*.
Part of what it enforces is how the nodes happen to be numbered.

### Local walls versus uniform ones

The reviewer settled the question left open last time, and settled it against my
model. Undecidability of totality is **complete** for its class — the hardness is
spread uniformly over the whole address space. Nothing makes some addresses
decidable and others not. Marking particular nodes as unsettleable is an *extra
assumption I added*; it isn't what undecidability gives you.

What undecidability gives you is opacity attached to the **form of the move**. A
join names an arbitrary set, exactly as a limit names an arbitrary fundamental
sequence. If certifying that is a search that cannot conclude, then it cannot
conclude for *any* join — all of them at once, with nothing of that kind left to
detour to.

Measured, with a policy that can see the walls (40 steps, joins certified):

- **no wall** — 40
- **one address marked unsettleable** — 40, at every placement, for 6 and 10
  roots. The class survives; only the pairs touching that address die.
- **the form marked unsettleable** — **0**, at every root count.

### And the climb doesn't stop

Rank under form opacity is **identical** to rank with no wall at all, with zero
refusals — a depth-following policy never attempts a join, so nothing is ever
refused. Meanwhile a joining policy has 40 of them available and certifies none.

**What a uniform epistemic wall costs is not height. It is a kind of content.**
The system climbs forever and never certifies the one move that would join two
things it knows independently. That is §7's `hidden` verdict again, in a second
domain: complete about where it is, permanently unable to close one gap.

### A correction to §8h

The refusal counts in the previous section — "29 of 30 refused" — used a policy
that checks whether a join is *new* but not whether it is *allowed*, so it
re-offered a refused move forever. Under it, a local wall looks as destructive as
a uniform one. With a policy that can see the wall, the same configuration leaves
every join available.

**The wall didn't remove those moves; the policy deadlocked on them.** A count of
blocked moves measures the policy as much as the wall, whenever the policy can't
see the wall. The claims made with the depth-following policy — the 5-rung
detour, the halt under total opacity, the pricing invariance — are unaffected,
because that one does check.

### What it does not show

`opaque_form` is a **declared** uniform opacity. Nothing here proves any address
undecidable; it models what a system does when a whole class of move cannot be
certified.

And the instrument gives **two bits, not a taxonomy.** It separates price-readers
and numbering-readers from everything else. It does *not* distinguish a wall that
reads placement from any other wall that reads some magnitude — which is why
`max_arity`, which reads how many parents a move has, sits in the same column as
the opacity walls. Two perturbations, two bits.

---

## 9. What this does not say

Kept separate deliberately, because the results are aesthetically suggestive and
that is exactly when this section matters most.

**It does not say anything about experience or consciousness.** The measured
objects are formal presentations of arithmetic. The resemblance between
"complete about location, incomplete about necessity" and an interior report of
unboundedness is a **resemblance**. It is not evidence, and it was not tested.
There is also a thoroughly ordinary explanation for such reports — you cannot
observe your own discontinuity, so the endpoint was never in your data — and
that explanation and the mathematical one are, interestingly, *the same
structural fact* stated in two vocabularies. Neither licenses the other.

**It does not prove any undecidability result.** That totality is Π⁰₂-complete
and `O`-membership Π¹₁-complete are **cited theorems**. What was measured is only
that a bounded certifier cannot separate "total" from "total so far" — the
property that makes the citation apply.

**It does not measure proof-theoretic strength.** `Tₙ ⊬ Con(Tₙ)` is Gödel's
second theorem, discharged from stated assumptions. No run here could establish
it and none tried. The bounded proof search in the first experiment saturates at
about 20 formulas and is reported as a smoke test, explicitly not as evidence.

**Absolute symbol counts are not meaningful.** The proof predicate `Prf` is left
as a primitive symbol rather than expanded into arithmetic, so every count
carries the same unexpanded constant. Only *ratios between arms* are read, which
is why every headline here is a ratio.

**`I = n` is bookkeeping, not measurement.** The mathematical model puts the rank
at `ε₀ + n`. The programs carry a counter.

**And it does not establish that the generative-continuation claim is true in
general.** It shows that one concrete realisation of it can be given contingent
continuation — which the original could not, because it contained nothing that
could fail.

---

## 10. Check it yourself

```bash
python experiments/reflection_ladder.py          # naming vs listing
python experiments/reflection_capacity.py        # the budget, and r* = L/κ_max
python experiments/reflection_limits.py          # price becomes gate
python experiments/reflection_omega_squared.py   # limits of limits, and O's real cost
python experiments/reflection_interior.py        # what a theory knows about itself
python experiments/reflection_derived_g.py       # G derived rather than posited
python experiments/reflection_finite_box.py      # a ladder in a box
python experiments/reflection_model_ladder.py    # a ceiling nobody chose
python experiments/reflection_dag.py             # productive forever, going nowhere
python experiments/reflection_filters.py         # the walls select for sideways
python experiments/reflection_selection.py       # rank-aware selection
python experiments/reflection_cost_model.py      # description pricing, and a fake wall
python experiments/reflection_wall_audit.py      # what does each wall actually read?
```

Add `--quick` to any of them for a smoke run. Every experiment states its
predictions and their falsifiers in its own docstring *before* reporting
results, and prints its honest scope alongside its verdicts.

The machinery is `project_genesis/reflection.py`, `finite_ladder.py`,
`model_ladder.py` and `reflection_dag.py`; 285 tests across the thirteen
`tests/test_reflection_*.py` files run in about 4.8 seconds. Most of them are
adversarial rather than confirmatory — the claim that an added rule is a
*capability* rests entirely on the proof checker refusing malformed derivations,
and the claim that the interior view is exact rests on the prediction never
consulting the run.

---

## A note on method

Three separate bugs during this work had the same shape: a confident,
well-formed, precise-looking number that was the right answer to a *different
question*. A capacity scan whose horizon was shorter than the budget's own decay
time reported a threshold four orders of magnitude low. A wall-predictor that
checked obstacles in order of interest rather than order encountered named the
wrong obstacle. Neither crashed. Neither looked wrong.

Each was caught by the same thing: **a second, independent route to the number.**
A closed form to check the simulation against. A control arm that should behave
differently. A registered falsifier written down before the run.

That is the reason this document can distinguish what was measured from what was
assumed — not care, but redundancy. It is also why every experiment here carries
a deliberately broken arm. An instrument that cannot fail has not told you
anything when it succeeds.

Two later bugs had a different shape, and produced a second rule. A filter called
*epistemic* and a filter called *structural* were both partly reading size, and
both read correctly for eleven experiments — because nothing ever separated what
they were named from what they did. What separated them was **perturbing one
quantity and seeing which rules noticed**: swap the pricing, shift the numbering,
watch what moves.

So the standing rule for anything added after this: **classify a constraint by
the quantity it reads, not by the name it was given, and demonstrate the
classification.** `tests/test_reflection_wall_audit.py` enforces it — a new wall
that is not in the audited set fails a test. The cost is one line of
configuration; the alternative is carrying a wrong label indefinitely, which is
what happened twice.

A third rule falls out of the same run. **A count of blocked moves measures the
policy as much as the wall, whenever the policy cannot see the wall.** A
join-seeking policy that re-offered a refused move forever made a local
constraint look like a global one. The fix is not to trust a refusal count until
the policy proposing the moves is filter-aware — and both behaviours are pinned
in tests, so the artefact stays visible rather than being quietly corrected.
