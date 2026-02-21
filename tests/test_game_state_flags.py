#!/usr/bin/env python3
"""Unit tests for worker-state overlay flag parsing."""

import unittest

from game import _parse_update_flags


class TestGameStateFlags(unittest.TestCase):
    def test_won_state(self) -> None:
        update = {"args": [{"state": "won"}]}
        self.assertEqual(_parse_update_flags(update), (False, True))

    def test_game_over_state(self) -> None:
        update = {"args": [{"state": "game_over"}]}
        self.assertEqual(_parse_update_flags(update), (True, False))

    def test_playing_state_is_inconclusive(self) -> None:
        update = {"args": [{"state": "playing"}]}
        self.assertEqual(_parse_update_flags(update), (None, None))

    def test_missing_state(self) -> None:
        self.assertEqual(_parse_update_flags({}), (None, None))


if __name__ == "__main__":
    unittest.main()
