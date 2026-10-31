# Project Genesis, in plain words

*For anyone curious. No background needed. This says what the project is
trying to do, how it goes about it, what it has found, and what it has not.*

---

## The question

Some things last and some don't. A wave breaks; a crystal stays. A rumour
fades; a language lasts for centuries. In very different settings, structure
appears, and some of it persists.

This project asks: is there a simple rule for which structures last? And does
the same rule show up in systems that seem to have nothing to do with each
other?

## The idea being tested

The idea has two parts.

**Making differences is easy. Holding them together is hard.** A growing
system keeps producing new differences: boundaries, parts, distinctions. That
goes for a cooling material, a brain, or a body of knowledge. Binding those
parts into something that works as a whole costs more, and it costs most where
the most is going on. So a system can always make more than it can hold
together.

**What lasts is what can be held together, and it becomes the ground for what
comes next.** Each structure that holds becomes material for new ones. So the
range of what's possible isn't fixed in advance. It grows as the system grows.
On this view there is no final shape to arrive at, only the next stable step.

As a formula, the idea is `S = ΔC + κ·ΔI`. Distinction (`ΔC`) plus integration
(`ΔI`), with integration weighted by a limited capacity (`κ`). You don't need
the formula to follow the rest.

## How it's tested

An idea this broad is easy to state and hard to check. So the project turns it
into small, exact questions a computer can answer:

1. Write down a prediction, and what result would prove it wrong, *before*
   running anything.
2. Build a simulation or a mathematical model that can answer it.
3. Run it, and record the verdict, including when the prediction fails.
4. Compare against a baseline where nothing is going on, such as a random
   pattern, so that a result isn't credited to the idea when chance would do
   the same.

Failures stay on the record. Many predictions here have failed, and those
failures have often taught the most.

## What has been found

**Measured:**

- **Where regions meet.** A simulated material sorts itself into regions. The
  points where those regions meet involve one more region than there are
  dimensions: three in a flat sheet, four in space. This holds from one to four
  dimensions (weakly in four). But a random pattern does the same, so this is a
  fact about geometry, not evidence for the idea. What the simulation adds is
  that awkward meeting points, such as four regions crowded at one point in a
  flat sheet, break apart into simpler ones.
- **Scarcity pushes a system out of rigid order.** When holding structure
  together uses up a limited resource, a system's best state gets pushed out
  of rigid order. The condition for when that happens is the same in three
  unrelated model systems: the simulated material, a model of memory, and a
  model of things falling into rhythm together. Whether it happens at the same
  point in all three is not settled.
- **Edges make room.** Between two regions sits a border where neither fully
  holds. A third kind, held dormant everywhere else, first appears there. In a
  narrow range of settings it grows into those borders, and new three-way
  meeting points form. This is shown in two dimensions only, and the third kind
  was built into the model from the start.
- **A ladder with no top.** In mathematics, a theory can be extended by adding
  the statement "the theory below me is consistent", again and again. It has
  been known since Gödel that this ladder never reaches a top. The project
  measured what keeps a climber on it: being able to refer to things by name,
  at a cost that doesn't grow with the size of what's named.

**Not shown:**

- That the physical universe works this way. These are models, small and
  simplified, and the documents say so wherever it matters.
- That every field of knowledge is, at bottom, the same process. That is the
  larger hypothesis behind the project, not a result of it.
- That the two halves, the simulations and the mathematics, share one
  mechanism. They look alike. Whether they really are alike is the next open
  question.

## Where the idea has relatives

None of this comes from nowhere. Mathematicians describe the universe of sets
as built in stages, with no last stage. People who study evolution and
complexity speak of the "adjacent possible": each new thing opens
possibilities that didn't exist before. Physicists and chemists study how
stable patterns form and persist. Logicians since Turing have studied ladders
of theories that reason about the ones below them.

This project tries to add something modest to that shared conversation: one
rule, stated precisely enough to be tested, and tested in the open.

## Why it might matter

If the idea holds up, it could offer a common language for how structure forms
and lasts. People in different fields could check it against their own work
rather than take it on trust. That is an "if". In the meantime, the useful part
is the method: say what would prove you wrong, look, and keep the record either
way.

## Reading further

| If you want | Read |
|---|---|
| The idea in full, with every claim marked as measured, untested, or declined | [`The_Principle.md`](The_Principle.md) |
| One result that stands entirely on its own | [`Junction_Selection.md`](Junction_Selection.md) |
| Every experiment, including the failures | [`Experiment_Log.md`](Experiment_Log.md) |
| The mathematical half, without proof theory | [`The_Reflection_Ladder.md`](The_Reflection_Ladder.md) |
