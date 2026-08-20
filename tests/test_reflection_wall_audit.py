"""Tests for the wall audit: classifying walls by what they read.

The instrument perturbs one quantity at a time and asks which walls move. That
only means anything if four things hold:

1. **The identity perturbation must be a pure relabelling.** If shifting the
   first identifier changed the graph's behaviour on its own, the whole
   `reads_identity` column would be measuring a bug in `ReflectionGraph` rather
   than a property of a wall. This is the control the audit rests on and it is
   tested first.

2. **The classification must be pinned per wall**, including the two it
   convicts: `certify_effort` as a size tax, and `address_bits` as partly an
   encoding artifact — the second one found by the instrument rather than by a
   reviewer.

3. **Uniform and local opacity must be distinguishable**, and distinguishable
   with a policy that can see the filters. Under a filter-blind policy they look
   alike, which is exactly the confound the earlier run fell into, so both
   behaviours are pinned: the real separation and the artefact that hid it.

4. **Removing a move class must not be an economic wall in disguise.** Rank
   under form opacity has to equal rank with no wall at all, or the finding is
   just a cost by another name.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.reflection_dag import (
    Filters, ReflectionGraph, broaden, deepen, join_aware, joins, reflect,
    run_adaptive, run_filtered, run_policy,
)

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

from reflection_wall_audit import WALLS, audit, locality  # noqa: E402


# ------------------------------------------- 1. the control the audit rests on


class TestRelabellingIsPure(unittest.TestCase):
    """A shifted graph must be the same graph with different numbers on it."""

    def test_base_graph_is_isomorphic_under_a_shift(self):
        a = ReflectionGraph.base(roots=4)
        b = ReflectionGraph.base(roots=4, first_id=17)
        self.assertEqual(a.size, b.size)
        self.assertEqual(a.rank, b.rank)
        self.assertEqual([n.depth for n in a.nodes], [n.depth for n in b.nodes])
        self.assertEqual(b.node(17).content, frozenset({17}))
        self.assertEqual(b.node(20).content, frozenset({20}))

    def test_new_nodes_continue_from_the_offset(self):
        g = ReflectionGraph.base(roots=2, first_id=100)
        g = reflect(g, frozenset({100})).graph_after
        self.assertEqual(g.node(102).parents, frozenset({100}))
        self.assertEqual(g.node(102).depth, 1)

    def test_an_unfiltered_run_is_identical_under_a_shift(self):
        for policy in (deepen, broaden):
            a = run_filtered(policy, 30, first_id=0)
            b = run_filtered(policy, 30, first_id=8)
            self.assertEqual(a["tally"], b["tally"], policy)
            self.assertEqual(a["final_rank"], b["final_rank"], policy)
            self.assertEqual(a["joins"], b["joins"], policy)

    def test_rank_aware_is_identical_under_a_shift(self):
        a = run_adaptive(40, first_id=0)
        b = run_adaptive(40, first_id=8)
        self.assertEqual(a["tally"], b["tally"])
        self.assertEqual(a["final_rank"], b["final_rank"])

    def test_the_default_offset_is_zero(self):
        """So every earlier result keeps the numbering it was measured under."""
        self.assertEqual(ReflectionGraph.base(roots=3).first_id, 0)
        self.assertEqual(run_policy(deepen, 5, roots=2)["final_rank"],
                         run_filtered(deepen, 5, roots=2, warmup=0)["final_rank"])


# ------------------------------------------------- 2. the classification itself


class TestAudit(unittest.TestCase):

    EXPECTED = {"budget": (True, False), "certify_effort": (True, False),
                "address_bits": (True, True), "opaque": (False, False),
                "opaque_form": (False, False), "max_arity": (False, False)}

    def test_every_wall_in_the_module_is_audited(self):
        """A new wall must not be able to slip in unclassified."""
        self.assertEqual(set(WALLS), set(self.EXPECTED))

    def test_the_classification_is_what_was_registered(self):
        for row in audit(30):
            self.assertEqual((row["reads_price"], row["reads_identity"]),
                             self.EXPECTED[row["wall"]], row["wall"])

    def test_certify_effort_convicts_as_a_size_tax(self):
        rows = {r["wall"]: r for r in audit(30)}
        self.assertTrue(rows["certify_effort"]["reads_price"])
        self.assertEqual(rows["certify_effort"]["reads_price"],
                         rows["budget"]["reads_price"])

    def test_address_bits_changes_verdict_on_a_pure_relabel(self):
        """The second mislabelled wall, stated directly rather than through a
        run: the same structural key, renumbered, gets a different answer."""
        f = Filters(address_bits=3)
        small = frozenset({0, 1, 2})
        shifted = frozenset({8, 9, 10})
        self.assertTrue(f.admits(small, max(small), 1)[0])
        self.assertFalse(f.admits(shifted, max(shifted), 1)[0])

    def test_the_opacity_walls_read_neither_perturbation(self):
        rows = {r["wall"]: r for r in audit(30)}
        for wall in ("opaque", "opaque_form"):
            self.assertFalse(rows[wall]["reads_price"], wall)
            self.assertFalse(rows[wall]["reads_identity"], wall)


# ----------------------------------------------- 3. local against uniform


class TestOpaqueForm(unittest.TestCase):

    def test_it_refuses_by_arity_and_ignores_the_key(self):
        f = Filters(opaque_form="join")
        big = frozenset(range(500))
        self.assertEqual(f.admits(big, max(big), 1), (True, None))
        self.assertEqual(f.admits(frozenset({0, 1}), 1, 2),
                         (False, "uncertifiable"))

    def test_an_unknown_form_is_refused(self):
        with self.assertRaises(ValueError):
            Filters(opaque_form="vibes").admits(frozenset({1}), 1, 1)

    def test_it_is_off_by_default(self):
        self.assertIsNone(Filters().opaque_form)

    def test_uniform_opacity_removes_the_class_at_every_root_count(self):
        for row in locality([3, 6, 10], 40):
            self.assertEqual(row["form"], 0, row["roots"])

    def test_local_opacity_leaves_the_class_intact(self):
        """Except where the graph is so small that the marked root is in every
        incomparable pair — which is a fact about three roots, not about walls."""
        for row in locality([6, 10], 40):
            self.assertEqual(set(row["local"]), {row["clean"]}, row["roots"])

    def test_the_separation_survives_both_cost_models(self):
        for model in ("content", "description"):
            f = Filters(opaque_form="join", cost_model=model)
            self.assertEqual(
                run_filtered(join_aware(f), 40, filters=f)["joins"], 0, model)


class TestTheConfoundIsPinned(unittest.TestCase):
    """A filter-blind policy makes local opacity look like uniform opacity. That
    is what the earlier refusal counts measured, so it is pinned rather than
    quietly fixed."""

    def test_broaden_reoffers_a_refused_pair_forever(self):
        f = Filters(opaque=frozenset({0}))
        out = run_filtered(broaden, 30, roots=6, filters=f)
        self.assertEqual(out["joins"], 0)
        self.assertEqual(out["blocks"]["uncertifiable"], 30)

    def test_a_filter_aware_policy_finds_the_joins_that_are_there(self):
        f = Filters(opaque=frozenset({0}))
        out = run_filtered(join_aware(f), 30, roots=6, filters=f)
        self.assertGreater(out["joins"], 0)
        self.assertEqual(out["blocks"]["uncertifiable"], 0)

    def test_join_aware_never_proposes_an_inadmissible_join(self):
        for marked in range(6):
            f = Filters(opaque=frozenset({marked}))
            out = run_filtered(join_aware(f), 30, roots=6, filters=f)
            self.assertEqual(sum(out["blocks"].values()), 0, marked)

    def test_join_aware_falls_back_to_deepening_rather_than_stalling(self):
        f = Filters(opaque_form="join")
        out = run_filtered(join_aware(f), 30, roots=6, filters=f)
        self.assertEqual(out["joins"], 0)
        self.assertEqual(out["tally"]["advancing"] + out["tally"]["duplicate"],
                         30)


# ------------------------------------- 4. it is not an economic wall in disguise


class TestTheClimbSurvives(unittest.TestCase):

    def test_rank_is_untouched_by_form_opacity(self):
        for model in ("content", "description"):
            clean = run_adaptive(40, filters=Filters(cost_model=model))
            opaque = run_adaptive(40, filters=Filters(opaque_form="join",
                                                      cost_model=model))
            self.assertEqual(clean["final_rank"], opaque["final_rank"], model)
            self.assertEqual(opaque["blocks"]["uncertifiable"], 0, model)

    def test_what_is_lost_is_a_kind_of_content_not_height(self):
        clean = Filters()
        form = Filters(opaque_form="join")
        available = run_filtered(join_aware(clean), 40, filters=clean)["joins"]
        certified = run_filtered(join_aware(form), 40, filters=form)["joins"]
        self.assertGreater(available, 0)
        self.assertEqual(certified, 0)

    def test_joins_counts_multi_parent_nodes_only(self):
        g = ReflectionGraph.base(roots=3)
        g = reflect(g, frozenset({0})).graph_after
        self.assertEqual(joins(g), 0)
        g = reflect(g, frozenset({1, 2})).graph_after
        self.assertEqual(joins(g), 1)


if __name__ == "__main__":
    unittest.main()
