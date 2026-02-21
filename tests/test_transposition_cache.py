#!/usr/bin/env python3
"""Unit tests for transposition cache policy behavior."""

import unittest

from transposition_cache import TranspositionCache


class TestTranspositionCache(unittest.TestCase):
    def test_hit_miss_tracking(self) -> None:
        c = TranspositionCache(cap=4)
        c.store(("a", 0, 0), 1.5)
        self.assertEqual(c.get(("a", 0, 0)), 1.5)
        self.assertIsNone(c.get(("missing", 0, 0)))
        self.assertEqual(c.stats(), {"hits": 1, "misses": 1})

    def test_clear_on_cap_when_not_oversized_preload(self) -> None:
        c = TranspositionCache(cap=2)
        c.store(("k1", 0, 0), 1.0)
        c.store(("k2", 0, 0), 2.0)
        c.store(("k3", 0, 0), 3.0)
        self.assertEqual(len(c.table), 1)
        self.assertEqual(c.table[("k3", 0, 0)], 3.0)

    def test_oversized_preload_is_preserved(self) -> None:
        c = TranspositionCache(cap=2)
        c.load({
            ("a", 0, 0): 1.0,
            ("b", 0, 0): 2.0,
            ("c", 0, 0): 3.0,
        })
        self.assertTrue(c.keep_oversized_preload)
        before = len(c.table)
        c.store(("new", 0, 0), 4.0)
        self.assertEqual(len(c.table), before)
        self.assertIn(("new", 0, 0), c.new_entries)


if __name__ == "__main__":
    unittest.main()
