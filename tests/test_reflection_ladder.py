"""Tests for the reflection ladder generator.

Three things in `project_genesis/reflection.py` are load-bearing, and if any of
them is wrong the experiment's conclusions are worth nothing:

1. **The Gödel numbering is injective.** The ladder's productivity test is
   `Con(T_n) not in axioms(T_n)`, and the index that distinguishes one theory
   from another is a code. A colliding code would silently fuse distinct
   theories — which is exactly the failure the `truncated` arm is *built* to
   exhibit on purpose, so it had better not happen anywhere else by accident.

2. **The proof checker rejects bad proofs.** The claim that a rung is a
   capability rather than a stored string rests entirely on a machine-checked
   derivation. A checker that accepts anything proves nothing, so most of the
   tests below are adversarial: malformed derivations that must be refused.

3. **The truncated arm stalls exactly where the arithmetic says.** It is the
   negative control for the whole experiment. If it stalled for an incidental
   reason — or did not stall at all — the registered dissociation in Q3 would
   be an artefact.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.reflection import (
    And, CAPACITY, Eq, Exists, Forall, Implies, Line, Not, Num, Or, Plus, Prf,
    ProofError, Succ, Times, Var, Zero, check_proof, closure_search, code_bits,
    con_formula, conjoin_rungs_proof, first_index_collision, free_vars,
    godel_number, induction_instance, integration_rank, is_axiom,
    is_induction_instance, ladder, nominal_increment, pa_base_axioms, pair,
    peano, productive_increment, serialize, step, substitute, symbols,
)


X, Y, P = Var("x"), Var("y"), Var("p")


def _corpus():
    """A spread of syntactically distinct objects, including near-misses that
    a sloppy encoding would fuse."""
    return [
        Zero(), Succ(Zero()), Num(0), Num(1), Num(255), Num(256),
        Var("x"), Var("y"), Var("xy"), Var("x1"),
        Plus(X, Y), Plus(Y, X), Times(X, Y),
        Eq(X, Y), Eq(Y, X), Eq(Plus(X, Y), Zero()),
        Not(Eq(X, Y)), And(Eq(X, Y), Eq(Y, X)), Or(Eq(X, Y), Eq(Y, X)),
        Implies(Eq(X, Y), Eq(Y, X)),
        Forall("x", Eq(X, X)), Exists("x", Eq(X, X)), Forall("y", Eq(X, X)),
        Prf(Num(1), P, Num(2)), Prf(Num(2), P, Num(1)),
        *pa_base_axioms(),
    ]


# ------------------------------------------------------- 1. the coding is sane


class TestGodelNumbering(unittest.TestCase):

    def test_injective_on_a_corpus_of_distinct_objects(self):
        objects = _corpus()
        codes = [godel_number(o) for o in objects]
        self.assertEqual(len(set(codes)), len(objects),
                         "distinct syntax collided under the coding")

    def test_serialisation_is_prefix_free_across_concatenation(self):
        """`f(a,b)` and `f(a',b')` must not collide when the children's
        encodings could be split differently. This is the property that makes
        the sequence encoding decodable, and it is easy to lose."""
        a = And(Eq(Zero(), Zero()), Eq(Succ(Zero()), Zero()))
        b = And(Eq(Zero(), Succ(Zero())), Eq(Zero(), Zero()))
        self.assertNotEqual(serialize(a), serialize(b))
        self.assertNotEqual(godel_number(a), godel_number(b))

    def test_variable_names_of_different_length_do_not_fuse(self):
        self.assertNotEqual(godel_number(Var("ab")), godel_number(Var("a")))
        self.assertNotEqual(godel_number(Forall("ab", Eq(X, X))),
                            godel_number(Forall("a", Eq(X, X))))

    def test_sequence_coding_distinguishes_order_and_length(self):
        f, g = Eq(X, Y), Eq(Y, X)
        self.assertNotEqual(godel_number([f, g]), godel_number([g, f]))
        self.assertNotEqual(godel_number([f]), godel_number([f, f]))

    def test_leading_byte_keeps_the_map_injective(self):
        """Without the sentinel, encodings differing only in leading zero bytes
        would read back as the same integer."""
        for o in _corpus():
            self.assertGreater(godel_number(o).bit_length(),
                               8 * (len(serialize(o)) - 1))

    def test_pairing_is_injective(self):
        seen = {pair(a, b) for a in range(40) for b in range(40)}
        self.assertEqual(len(seen), 1600)

    def test_numerals_cost_their_bit_length(self):
        self.assertEqual(symbols(Num(0)), 1)
        self.assertEqual(symbols(Num(1)), 1)
        self.assertEqual(symbols(Num(2 ** 64)), 65)

    def test_code_bits_grows_with_content(self):
        small = code_bits(Eq(Zero(), Zero()))
        large = code_bits(Eq(Num(2 ** 4096), Zero()))
        self.assertGreater(large, small + 4000)


# ------------------------------------------------- 2. substitution and schemas


class TestSubstitution(unittest.TestCase):

    def test_substitutes_free_occurrences_only(self):
        f = And(Eq(X, Zero()), Forall("x", Eq(X, Succ(Zero()))))
        got = substitute(f, "x", Num(7))
        self.assertEqual(got.left, Eq(Num(7), Zero()))
        self.assertEqual(got.right, f.right, "bound occurrence was captured")

    def test_avoids_capture_by_renaming(self):
        """Substituting `y` into `∀y φ(x)` must rename the bound `y`."""
        f = Forall("y", Eq(X, Y))
        got = substitute(f, "x", Y)
        self.assertNotEqual(got.var, "y")
        self.assertIn("y", free_vars(got))

    def test_free_vars_respects_binders(self):
        self.assertEqual(free_vars(Forall("x", Eq(X, Y))), frozenset({"y"}))
        self.assertEqual(free_vars(Exists("p", Prf(Num(1), P, Num(2)))),
                         frozenset())

    def test_induction_instance_round_trips(self):
        for phi in (Eq(Plus(X, Zero()), X),
                    Not(Eq(Succ(X), Zero())),
                    Exists("y", Eq(Plus(X, Y), Zero()))):
            self.assertTrue(is_induction_instance(induction_instance(phi, "x")))

    def test_induction_check_rejects_near_misses(self):
        """A checker that accepts these would let anything in under `axiom`."""
        phi = Eq(Plus(X, Zero()), X)
        good = induction_instance(phi, "x")
        bad = [
            Eq(Zero(), Zero()),
            # base case replaced by something else
            Implies(And(Eq(Zero(), Zero()), good.left.right), good.right),
            # step case concluding φ(x) rather than φ(Sx)
            Implies(And(good.left.left, Forall("x", Implies(phi, phi))),
                    good.right),
            # conclusion generalises the wrong variable
            Implies(good.left, Forall("y", phi)),
            # inductive hypothesis is not φ
            Implies(And(good.left.left,
                        Forall("x", Implies(Eq(X, X),
                                            substitute(phi, "x", Succ(X))))),
                    good.right),
        ]
        for f in bad:
            self.assertFalse(is_induction_instance(f), f"accepted: {f}")


# ------------------------------------------------------ 3. the checker refuses


class TestProofChecker(unittest.TestCase):

    def setUp(self):
        self.theory = list(ladder(peano("indexed"), 3))[-1].theory_after

    def test_accepts_a_genuine_derivation(self):
        proof = conjoin_rungs_proof(self.theory)
        conclusion = check_proof(self.theory, proof)
        self.assertIsInstance(conclusion, And)
        self.assertGreaterEqual(len(proof), 3 * (len(self.theory.rungs) - 1))

    def test_every_rung_is_actually_used_as_a_premise(self):
        """The conjunction must mention each rung — otherwise the derivation
        would 'use' the axioms only by listing them."""
        proof = conjoin_rungs_proof(self.theory)
        conclusion = check_proof(self.theory, proof)
        flat, stack = [], [conclusion]
        while stack:
            node = stack.pop()
            if isinstance(node, And):
                stack.extend([node.left, node.right])
            else:
                flat.append(node)
        for rung in self.theory.rungs:
            self.assertIn(rung, flat)

    def test_rejects_a_non_axiom_claimed_as_an_axiom(self):
        alien = con_formula(self.theory)  # true of T_4, not an axiom of T_3
        with self.assertRaises(ProofError):
            check_proof(self.theory, [Line(alien, "axiom")])

    def test_rejects_another_theorys_rung(self):
        other = list(ladder(peano("inline"), 3))[-1].theory_after
        with self.assertRaises(ProofError):
            check_proof(self.theory, [Line(other.rungs[0], "axiom")])

    def test_rejects_bad_modus_ponens(self):
        a, b = self.theory.rungs[0], self.theory.rungs[1]
        bad = [Line(a, "axiom"), Line(b, "axiom"),
               Line(And(a, b), "mp", (0, 1))]
        with self.assertRaises(ProofError):
            check_proof(self.theory, bad)

    def test_rejects_forward_references(self):
        a = self.theory.rungs[0]
        with self.assertRaises(ProofError):
            check_proof(self.theory, [Line(a, "mp", (1, 2)), Line(a, "axiom"),
                                      Line(a, "axiom")])

    def test_rejects_a_bogus_logical_schema(self):
        a, b = self.theory.rungs[0], self.theory.rungs[1]
        with self.assertRaises(ProofError):
            check_proof(self.theory, [Line(Implies(a, b), "logical")])

    def test_accepts_the_real_logical_schemas(self):
        a, b = self.theory.rungs[0], self.theory.rungs[1]
        for f in (Implies(a, Implies(b, a)),
                  Implies(And(a, b), a),
                  Implies(And(a, b), b),
                  Implies(a, Implies(b, And(a, b))),
                  Implies(a, Or(a, b)),
                  Implies(Implies(Not(a), Not(b)), Implies(b, a))):
            self.assertEqual(check_proof(self.theory, [Line(f, "logical")]), f)

    def test_rejects_bad_generalisation_and_instantiation(self):
        a = self.theory.rungs[0]
        with self.assertRaises(ProofError):
            check_proof(self.theory, [Line(a, "axiom"),
                                      Line(Forall("z", Eq(X, X)), "gen",
                                           (0,), var="z")])
        with self.assertRaises(ProofError):
            check_proof(self.theory,
                        [Line(Forall("x", Eq(X, X)), "axiom"),
                         Line(Eq(Zero(), Zero()), "ui", (0,), term=Zero())])

    def test_accepts_universal_instantiation_of_a_real_axiom(self):
        axiom = pa_base_axioms()[2]          # ∀x (x + 0 = x)
        proof = [Line(axiom, "axiom"),
                 Line(Eq(Plus(Num(3), Zero()), Num(3)), "ui", (0,),
                      term=Num(3))]
        self.assertEqual(check_proof(self.theory, proof).left.left, Num(3))

    def test_rejects_unknown_rules_and_empty_proofs(self):
        with self.assertRaises(ProofError):
            check_proof(self.theory, [])
        with self.assertRaises(ProofError):
            check_proof(self.theory, [Line(self.theory.rungs[0], "wishing")])

    def test_induction_instances_are_axioms_of_a_theory_with_the_schema(self):
        inst = induction_instance(Eq(Plus(X, Zero()), X), "x")
        self.assertTrue(is_axiom(self.theory, inst))
        self.assertEqual(check_proof(self.theory, [Line(inst, "axiom")]), inst)


# ------------------------------------------------------------ 4. the ladder


class TestLadder(unittest.TestCase):

    def test_con_names_the_theory_being_reflected_on(self):
        """`Con(T_n)` must carry `T_n`'s index — not `T_{n+1}`'s, which would
        make the ladder self-referential in a way the construction does not
        license."""
        t = peano("indexed")
        s = step(t)
        self.assertIsInstance(s.con, Not)
        atom = s.con.arg.body
        self.assertIsInstance(atom, Prf)
        self.assertEqual(atom.theory, Num(t.index()))
        self.assertNotEqual(atom.theory, Num(s.theory_after.index()))

    def test_con_asserts_unprovability_of_falsity(self):
        con = con_formula(peano("indexed"))
        self.assertIsInstance(con, Not)
        self.assertIsInstance(con.arg, Exists)
        self.assertEqual(con.arg.body.sentence,
                         Num(godel_number(Eq(Zero(), Succ(Zero())))))

    def test_every_rung_is_productive_in_the_honest_presentations(self):
        for kind in ("inline", "indexed"):
            steps = list(ladder(peano(kind), 10))
            self.assertTrue(all(s.new_axiom for s in steps), kind)
            self.assertTrue(all(productive_increment(s) == 1 for s in steps))
            self.assertIsNone(first_index_collision(steps), kind)

    def test_axiom_count_tracks_productive_rungs(self):
        steps = list(ladder(peano("indexed"), 7))
        base = len(pa_base_axioms())
        for s in steps:
            self.assertEqual(len(s.theory_after.axioms()),
                             base + sum(1 for t in steps[:s.n + 1]
                                        if t.new_axiom))

    def test_truncated_arm_stalls_exactly_at_the_wrap(self):
        for width in (2, 3, 4):
            wrap = 1 << width
            steps = list(ladder(peano("truncated", width=width), wrap + 3))
            for s in steps[:wrap]:
                self.assertTrue(s.new_axiom, f"width {width} rung {s.n}")
            for s in steps[wrap:]:
                self.assertFalse(s.new_axiom, f"width {width} rung {s.n}")
            self.assertEqual(first_index_collision(steps), wrap)

    def test_truncated_arm_freezes_its_axioms_while_the_rank_climbs(self):
        """The whole point of the negative control: I keeps counting, the
        theory does not move."""
        steps = list(ladder(peano("truncated", width=3), 12))
        final = steps[-1].theory_after
        at_wrap = steps[8].theory_before
        self.assertEqual(final.axioms(), at_wrap.axioms())
        self.assertEqual(integration_rank(final), 12)
        self.assertEqual(integration_rank(at_wrap), 8)
        self.assertEqual(nominal_increment(final), 1)
        self.assertEqual(sum(productive_increment(s) for s in steps[8:]), 0)

    def test_the_two_honest_arms_are_different_theories(self):
        """They are the same construction under two presentations, not the
        same theory — the experiment's wording depends on this."""
        a = list(ladder(peano("inline"), 3))[-1].theory_after
        b = list(ladder(peano("indexed"), 3))[-1].theory_after
        self.assertNotEqual(a.rungs, b.rungs)
        self.assertNotEqual(a.index(), b.index())
        self.assertEqual(len(a.axioms()), len(b.axioms()))

    def test_readding_an_existing_sentence_does_not_grow_the_axiom_set(self):
        t = peano("truncated", width=1)
        steps = list(ladder(t, 4))
        counts = [len(s.theory_after.axioms()) for s in steps]
        self.assertEqual(counts, [9, 10, 10, 10])

    def test_peano_validates_its_arguments(self):
        with self.assertRaises(ValueError):
            peano("hearsay")
        with self.assertRaises(ValueError):
            peano("truncated")
        with self.assertRaises(ValueError):
            Num(-1)


# ------------------------------------------------------------------ 5. cost


class TestCostSeparation(unittest.TestCase):
    """Q2's numbers, asserted rather than eyeballed."""

    def test_inline_presentation_grows_geometrically(self):
        steps = list(ladder(peano("inline"), 10))
        sizes = [s.theory_after.presentation_symbols() for s in steps]
        ratios = [b / a for a, b in zip(sizes, sizes[1:])]
        self.assertTrue(all(r > 1.9 for r in ratios[3:]), ratios)
        self.assertLess(ratios[-1], 2.2)

    def test_indexed_presentation_is_flat(self):
        steps = list(ladder(peano("indexed"), 10))
        sizes = {s.theory_after.presentation_symbols() for s in steps}
        self.assertEqual(len(sizes), 1, "recursive presentation should not grow")

    def test_indexed_expanded_cost_is_linear(self):
        steps = list(ladder(peano("indexed"), 10))
        sizes = [s.theory_after.expanded_symbols() for s in steps]
        deltas = {b - a for a, b in zip(sizes, sizes[1:])}
        self.assertEqual(len(deltas), 1, "listing rungs should cost a constant")

    def test_cost_separation_at_equal_productive_content(self):
        n = 12
        inline = list(ladder(peano("inline"), n))[-1].theory_after
        indexed = list(ladder(peano("indexed"), n))[-1].theory_after
        self.assertEqual(len(inline.rungs), len(indexed.rungs))
        ratio = (inline.presentation_symbols()
                 / indexed.presentation_symbols())
        self.assertGreater(ratio, 100.0)

    def test_capacity_is_a_constant_not_a_computation(self):
        self.assertIsInstance(CAPACITY, str)
        for kind in ("inline", "indexed"):
            for s in ladder(peano(kind), 3):
                self.assertLess(integration_rank(s.theory_after), float("inf"))


# ----------------------------------------------------- 6. the search is honest


class TestClosureSearch(unittest.TestCase):

    def test_finds_a_target_that_needs_modus_ponens(self):
        """Calibration. A negative result from a search that cannot find
        anything is not evidence of anything."""
        t = list(ladder(peano("indexed"), 4))[-1].theory_after
        a, b = t.rungs[0], t.rungs[1]
        target = And(a, b)
        self.assertNotIn(target, t.axioms())
        found, _ = closure_search(t, target, budget=20000,
                                  seeds=[Implies(a, Implies(b, target))])
        self.assertTrue(found)

    def test_does_not_find_the_next_consistency_sentence(self):
        t = list(ladder(peano("indexed"), 4))[-1].theory_after
        found, explored = closure_search(t, con_formula(t), budget=20000)
        self.assertFalse(found)
        self.assertLess(explored, 20000,
                        "closure should saturate, so the negative is "
                        "rule-limited rather than budget-limited")

    def test_respects_its_budget(self):
        t = list(ladder(peano("indexed"), 4))[-1].theory_after
        _, explored = closure_search(t, Eq(Num(99), Num(98)), budget=5)
        self.assertLessEqual(explored, 20)


if __name__ == "__main__":
    unittest.main()
