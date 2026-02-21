"""Transposition cache for strategy score_board lookups."""

from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass
class TranspositionCache:
    cap: int = 500_000
    table: OrderedDict = field(default_factory=OrderedDict)
    new_entries: dict = field(default_factory=dict)
    hits: int = 0
    misses: int = 0
    keep_oversized_preload: bool = False

    def load(self, entries: dict) -> None:
        """Bulk-load precomputed entries."""
        self.table.update(entries)
        self.keep_oversized_preload = len(self.table) > self.cap

    def get(self, key):
        if key not in self.table:
            self.misses += 1
            return None
        # LRU touch on successful read.
        self.table.move_to_end(key, last=True)
        val = self.table[key]
        self.hits += 1
        return val

    def store(self, key, value: float) -> None:
        """Store result and track newly added entries for DB flush."""
        if key in self.table:
            # Update existing entry and mark it as most recently used.
            self.table[key] = value
            self.table.move_to_end(key, last=True)
            self.new_entries[key] = value
            return

        if len(self.table) >= self.cap:
            if self.keep_oversized_preload:
                # Preserve huge preloads: do not insert more in-memory keys.
                self.new_entries[key] = value
                return
            # Evict exactly one least-recently-used entry.
            self.table.popitem(last=False)
        self.table[key] = value
        self.new_entries[key] = value

    def drain_new_entries(self) -> dict:
        out = dict(self.new_entries)
        self.new_entries.clear()
        return out

    def reset_stats(self) -> None:
        self.hits = 0
        self.misses = 0

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses}
