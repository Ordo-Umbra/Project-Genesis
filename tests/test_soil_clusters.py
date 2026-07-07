"""Checks for the fertile-soil percolation cluster analysis."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_genesis.soil_clusters import (  # noqa: E402
    SITE_PERCOLATION_PC,
    SITE_PERCOLATION_PC_3D,
    cluster_report,
    cluster_sizes,
    cluster_spans,
    label_periodic,
    largest_cluster_fraction,
    percolation_susceptibility,
    system_spans,
)


class TestLabelPeriodic(unittest.TestCase):

    def test_full_lattice_is_one_cluster(self):
        labels, n = label_periodic(np.ones((5, 5), bool))
        self.assertEqual(n, 1)
        self.assertTrue((labels == 1).all())

    def test_empty_lattice_has_no_clusters(self):
        labels, n = label_periodic(np.zeros((4, 4), bool))
        self.assertEqual(n, 0)
        self.assertTrue((labels == 0).all())

    def test_two_separate_blocks(self):
        m = np.zeros((6, 6), bool)
        m[0:2, 0:2] = True
        m[3:5, 3:5] = True
        labels, n = label_periodic(m)
        self.assertEqual(n, 2)
        self.assertEqual(sorted(cluster_sizes(labels, n).tolist()), [4, 4])

    def test_periodic_wrap_joins_across_edge(self):
        # single occupied column at c=0 and c=W-1 are periodic neighbours
        m = np.zeros((4, 4), bool)
        m[:, 0] = True
        m[:, 3] = True
        labels, n = label_periodic(m)
        self.assertEqual(n, 1)  # the two columns wrap into one cluster

    def test_diagonal_touch_is_not_connected(self):
        # 4-connectivity: diagonal neighbours are separate clusters
        m = np.zeros((5, 5), bool)
        m[1, 1] = True
        m[2, 2] = True
        _, n = label_periodic(m)
        self.assertEqual(n, 2)

    def test_rejects_unsupported_dim(self):
        with self.assertRaises(ValueError):
            label_periodic(np.ones((2, 2, 2, 2), bool))  # 4-D unsupported
        with self.assertRaises(ValueError):
            label_periodic(np.ones(5, bool))  # 1-D unsupported


class TestLabelPeriodic3D(unittest.TestCase):

    def test_full_cube_is_one_cluster(self):
        labels, n = label_periodic(np.ones((3, 3, 3), bool))
        self.assertEqual(n, 1)
        self.assertTrue((labels == 1).all())

    def test_two_separate_3d_blocks(self):
        m = np.zeros((6, 6, 6), bool)
        m[0:2, 0:2, 0:2] = True
        m[3:5, 3:5, 3:5] = True
        labels, n = label_periodic(m)
        self.assertEqual(n, 2)
        self.assertEqual(sorted(cluster_sizes(labels, n).tolist()), [8, 8])

    def test_periodic_wrap_in_3d(self):
        # occupied faces at z=0 and z=W-1 wrap into one cluster
        m = np.zeros((4, 4, 4), bool)
        m[:, :, 0] = True
        m[:, :, 3] = True
        _, n = label_periodic(m)
        self.assertEqual(n, 1)

    def test_face_diagonal_not_connected_in_3d(self):
        # 6-connectivity: cells touching only at an edge/corner stay separate
        m = np.zeros((5, 5, 5), bool)
        m[1, 1, 1] = True
        m[2, 2, 2] = True
        _, n = label_periodic(m)
        self.assertEqual(n, 2)

    def test_full_plane_spans_in_3d(self):
        # a cluster occupying an entire z-plane touches every x and every y
        m = np.zeros((4, 4, 4), bool)
        m[:, :, 2] = True
        labels, n = label_periodic(m)
        self.assertTrue(system_spans(labels, n))

    def test_small_3d_blob_does_not_span(self):
        m = np.zeros((5, 5, 5), bool)
        m[1:3, 1:3, 1:3] = True
        labels, n = label_periodic(m)
        self.assertFalse(system_spans(labels, n))

    def test_report_on_3d_mask(self):
        m = np.zeros((4, 4, 4), bool)
        m[0:2, 0:2, 0:2] = True  # 8 of 64 sites
        rep = cluster_report(m)
        self.assertEqual(rep["n_clusters"], 1)
        self.assertAlmostEqual(rep["p_inf"], 8 / 64)
        self.assertFalse(rep["spans"])

    def test_pc_3d_constant_is_simple_cubic_value(self):
        self.assertAlmostEqual(SITE_PERCOLATION_PC_3D, 0.3116, places=4)


class TestSpanning(unittest.TestCase):

    def test_full_row_spans(self):
        m = np.zeros((4, 4), bool)
        m[2, :] = True
        labels, n = label_periodic(m)
        self.assertTrue(cluster_spans(labels, 1))
        self.assertTrue(system_spans(labels, n))

    def test_full_column_spans(self):
        m = np.zeros((4, 4), bool)
        m[:, 1] = True
        labels, n = label_periodic(m)
        self.assertTrue(system_spans(labels, n))

    def test_small_blob_does_not_span(self):
        m = np.zeros((5, 5), bool)
        m[1:3, 1:3] = True
        labels, n = label_periodic(m)
        self.assertFalse(system_spans(labels, n))


class TestPercolationObservables(unittest.TestCase):

    def test_largest_cluster_fraction_full(self):
        self.assertEqual(largest_cluster_fraction(np.ones((4, 4), bool)), 1.0)

    def test_largest_cluster_fraction_empty(self):
        self.assertEqual(largest_cluster_fraction(np.zeros((4, 4), bool)), 0.0)

    def test_largest_cluster_fraction_partial(self):
        m = np.zeros((4, 4), bool)
        m[0:2, 0:2] = True  # 4 sites out of 16
        self.assertAlmostEqual(largest_cluster_fraction(m), 0.25)

    def test_susceptibility_excludes_largest(self):
        # sizes 10, 3, 2 -> finite {3,2}: (9+4)/(3+2)=2.6
        self.assertAlmostEqual(
            percolation_susceptibility(np.array([10, 3, 2])), 2.6, places=10)

    def test_susceptibility_single_cluster_is_zero(self):
        self.assertEqual(percolation_susceptibility(np.array([7])), 0.0)

    def test_susceptibility_empty_is_zero(self):
        self.assertEqual(percolation_susceptibility(np.array([])), 0.0)

    def test_report_keys_and_values(self):
        m = np.zeros((4, 4), bool)
        m[0:2, 0:2] = True
        rep = cluster_report(m)
        self.assertEqual(rep["n_clusters"], 1)
        self.assertAlmostEqual(rep["fertile_fraction"], 0.25)
        self.assertAlmostEqual(rep["p_inf"], 0.25)
        self.assertFalse(rep["spans"])

    def test_pc_constant_is_square_lattice_value(self):
        self.assertAlmostEqual(SITE_PERCOLATION_PC, 0.5927, places=4)

    def test_inputs_not_mutated(self):
        m = np.ones((3, 3), bool)
        before = m.copy()
        cluster_report(m)
        self.assertTrue((m == before).all())


if __name__ == "__main__":
    unittest.main()
