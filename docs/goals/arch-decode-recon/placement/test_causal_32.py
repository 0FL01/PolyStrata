#!/usr/bin/env python3
"""Synthetic checks for the forecast; never reads either evidence trace."""

import json
import unittest

import numpy as np

from causal_32 import CONFIG, CONFIG_SHA, Forecast, features, sha, similarity
from simulate import E, V


def synthetic(length):
    row = np.arange(8)
    route = np.zeros((length, V, 8), dtype=int)
    for t in range(length):
        for v in range(V):
            route[t, v] = (t + v + row) % E
    return route


class CausalTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads(CONFIG.read_text())

    def test_frozen_configuration(self):
        self.assertEqual(sha(CONFIG), CONFIG_SHA)

    def test_probability_and_prefix_only(self):
        train = synthetic(128)
        data = synthetic(256)
        t = 128
        f = Forecast(train, self.cfg)
        donor, w = f.predict(data[:t])
        self.assertEqual(w.shape, (32, 192))
        np.testing.assert_allclose(w.sum(axis=1), 1)
        np.testing.assert_allclose(w[:, :64].sum(axis=1), .8)
        self.assertTrue((w > 0).all())
        np.testing.assert_array_equal(donor[:64], data[t - 64:t])
        np.testing.assert_array_equal(donor[64:], train)
        self.assertFalse(np.allclose(w[0], w[-1]))
        data[t:] += E // 2  # poison current/future routes, leave prefix untouched
        donor_2, w_2 = f.predict(data[:t])
        np.testing.assert_array_equal(donor_2, donor)
        np.testing.assert_array_equal(w_2, w)

    def test_intersections_same_expert_in_same_physical_core_layer(self):
        repeat = np.tile(np.arange(8), (1, V, 1))
        disjoint = repeat.copy()
        disjoint[:, 18:30] += 8
        row, shared = features(repeat)
        broken_row, broken = features(disjoint)
        self.assertEqual(shared.sum(), 3 * 12 * 8)
        self.assertEqual(broken.sum(), 1 * 12 * 8)
        actual = similarity((np.concatenate((row, broken_row)), np.concatenate((shared, broken))),
                            (row[0], shared[0]), .5)
        np.testing.assert_allclose(actual, [1., .625])


if __name__ == "__main__":
    unittest.main()
