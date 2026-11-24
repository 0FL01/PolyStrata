#!/usr/bin/env python3
"""CPU unit tests for the placement counterfactual (not RDNA2 evidence)."""

import unittest

import numpy as np

from simulate import E, N, Objective, as_mask, joint_counts, misses_for, paid_replay


class PlacementTests(unittest.TestCase):
    def test_physical_ids_and_nonlinear_marginal_utility(self):
        r = np.array([[[0, 1, 2, 3, 4, 5, 6, 7], [E, 1 + E, 2 + E, 3 + E, 4 + E, 5 + E, 6 + E, 7 + E]]])
        # Joining only when there are more than seven misses: the second hit
        # on a visit is worthless; two hits should go to different layers.
        c = np.tile([0., 0., 0., 0., 0., 0., 0., 0., 10.], (2, 1))
        o = Objective(r, c)
        chosen = o.greedy(np.array([2]), [np.arange(N)])
        self.assertEqual(np.flatnonzero(chosen).tolist(), [0, E])
        initial = as_mask([0, 1])
        swap, = o.swaps(initial, [np.arange(N)], limit=1)
        self.assertEqual(swap, (E, 0, 10.))
        self.assertEqual(misses_for(r, initial).tolist(), [[7, 7]])

    def test_same_visit_swap_no_gain_if_both_experts_requested(self):
        r = np.array([[[0, 1, 2, 3, 4, 5, 6, 7]]])
        c = np.array([[0., 0., 0., 0., 0., 0., 2., 5., 6.]])
        o = Objective(r, c)
        self.assertEqual(o.swaps(as_mask([0]), [np.arange(8)], limit=1), [])

    def test_paid_copy_not_visible_until_completion_boundary(self):
        r = np.tile(np.arange(1, 9), (4, 48, 1))
        times = {"small": {"p90_ms": .2}, "pinned": {"p90_ms": 1.5, "bytes": 100}}
        m, cost, copied = paid_replay(r, 0, np.arange(4.), as_mask([0]), [], np.full(N, 100), 1,
                                      times, "pinned", lambda t, s, b: [(1, 0)] if t == 0 else [])
        self.assertEqual(m[:, 0].tolist(), [8, 8, 7, 7])
        self.assertEqual(cost.tolist(), [1.5, 0., 0., 0.])
        self.assertEqual(copied.tolist(), [100, 0, 0, 0])

    def test_joint_counts_same_physical_layer(self):
        m = np.zeros((1, 48), dtype=int)
        m[0, 6], m[0, 18], m[0, 30] = 2, 3, 4
        joint = joint_counts(m)
        self.assertEqual(joint['1->2'][2][3], 1)
        self.assertEqual(joint['2->3'][3][4], 1)
        self.assertEqual(joint['1->3'][2][4], 1)
        self.assertEqual(joint['1->3'][0][0], 11)


if __name__ == "__main__":
    unittest.main()
