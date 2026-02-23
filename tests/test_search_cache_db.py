#!/usr/bin/env python3
"""Tests for persisted search transposition cache rows in SQLite."""

import tempfile
import unittest
from pathlib import Path

import cache as db
from strategy_core import board_to_bb


class TestSearchCacheDB(unittest.TestCase):
    def setUp(self) -> None:
        self._old_db_path = db.DB_PATH
        self._old_schema_ready = db._SCHEMA_READY
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_path = str(Path(self._tmpdir.name) / "transposition.db")
        db.DB_PATH = self._db_path
        db._SCHEMA_READY = False

    def tearDown(self) -> None:
        db.DB_PATH = self._old_db_path
        db._SCHEMA_READY = self._old_schema_ready
        self._tmpdir.cleanup()

    def test_save_and_load_search_entries_by_version_tuple(self) -> None:
        b1 = board_to_bb([
            [512, 256, 128, 64],
            [32, 16, 8, 4],
            [2, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        b2 = board_to_bb([
            [1024, 512, 256, 128],
            [64, 32, 16, 8],
            [4, 2, 0, 0],
            [0, 0, 0, 0],
        ])
        entries = {
            (b1, 2, 1, 0, 4, 1): 123.5,
            (b2, 2, 0, 1, 5, 0): 456.25,
        }

        written = db.save_search_entries(entries, eval_version="1.7", search_version="s1")
        self.assertEqual(written, 2)

        loaded = db.load_search_version("1.7", "s1")
        self.assertEqual(len(loaded), 2)
        self.assertAlmostEqual(loaded[(b1, 2, 1, 0, 4, 1)], 123.5)
        self.assertAlmostEqual(loaded[(b2, 2, 0, 1, 5, 0)], 456.25)

        # Version tuple mismatch should not return rows.
        self.assertEqual(db.load_search_version("1.7", "s2"), {})
        self.assertEqual(db.load_search_version("1.8", "s1"), {})

    def test_load_search_entries_by_max_tile_range(self) -> None:
        b_low = board_to_bb([
            [256, 128, 64, 32],
            [16, 8, 4, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        b_high = board_to_bb([
            [2048, 1024, 512, 256],
            [128, 64, 32, 16],
            [8, 4, 2, 0],
            [0, 0, 0, 0],
        ])
        db.save_search_entries(
            {
                (b_low, 2, 0, 0, 4, 1): 111.0,
                (b_high, 2, 0, 0, 4, 1): 222.0,
            },
            eval_version="1.7",
            search_version="s1",
        )

        low_only = db.load_search_version("1.7", "s1", max_max_tile=512)
        self.assertEqual(set(low_only.keys()), {(b_low, 2, 0, 0, 4, 1)})

        high_only = db.load_search_version("1.7", "s1", min_max_tile=1024)
        self.assertIn((b_high, 2, 0, 0, 4, 1), high_only)
        self.assertNotIn((b_low, 2, 0, 0, 4, 1), high_only)

    def test_list_search_versions(self) -> None:
        b = board_to_bb([
            [128, 64, 32, 16],
            [8, 4, 2, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        db.save_search_entries({(b, 2, 0, 0, 3, 1): 77.0}, "1.7", "s1")
        db.save_search_entries({(b, 2, 0, 0, 4, 0): 88.0}, "1.7", "s2")

        versions = db.list_search_versions()
        self.assertEqual(versions.get(("1.7", "s1")), 1)
        self.assertEqual(versions.get(("1.7", "s2")), 1)


if __name__ == "__main__":
    unittest.main()
