"""
test_powers.py — Quick sanity check that swap and delete actually work in Playwright.

Makes a few moves to build a board, then:
  1. Forces a SWAP of the two largest tiles
  2. Forces a DELETE of the smallest tile value present
Prints the board before/after each to confirm the action took effect.
"""

import asyncio
from game import launch_browser, read_state, execute_move, execute_swap, execute_delete, print_board


async def main():
    print("Launching browser (headless)…")
    pw, browser, page = await launch_browser(headless=True)

    try:
        # Build up a non-trivial board with some moves — play until we see a 512
        print("\nPlaying until 512 tile appears (to unlock delete)…")
        from strategy import best_action
        for _ in range(500):
            s = await read_state(page)
            if s.over:
                break
            if any(s.board[r][c] >= 512 for r in range(4) for c in range(4)):
                break
            action = best_action(s.board, {}, depth=3)
            if action is None:
                break
            await execute_move(page, action[1])

        state = await read_state(page)
        print("\nBoard before any power-up:")
        print_board(state)
        print(f"Powers: {state.powers}")

        board = state.board

        # ── TEST SWAP ─────────────────────────────────────────────────────────
        # Find the two largest distinct-valued tiles
        tiles = sorted(
            ((board[r][c], r, c) for r in range(4) for c in range(4) if board[r][c] > 0),
            reverse=True,
        )
        # Pick two tiles with different values
        t1 = tiles[0]
        t2 = next((t for t in tiles[1:] if t[0] != t1[0]), None)

        if t2 and state.powers.get("swap", 0) > 0:
            v1, r1, c1 = t1
            v2, r2, c2 = t2
            print(f"\n=== SWAP TEST ===")
            print(f"Swapping {v1}@({r1},{c1}) <-> {v2}@({r2},{c2})")
            await execute_swap(page, r1, c1, r2, c2)
            state2 = await read_state(page)
            print("Board after swap:")
            print_board(state2)
            got1 = state2.board[r1][c1]
            got2 = state2.board[r2][c2]
            if got1 == v2 and got2 == v1:
                print(f"  SWAP OK: ({r1},{c1})={got1}, ({r2},{c2})={got2}")
            else:
                print(f"  SWAP FAILED: ({r1},{c1})={got1} (expected {v2}), ({r2},{c2})={got2} (expected {v1})")
            board = state2.board
        else:
            if state.powers.get("swap", 0) == 0:
                print("\n=== SWAP TEST SKIPPED (no swap uses remaining) ===")
            else:
                print("\n=== SWAP TEST SKIPPED (couldn't find two distinct tiles) ===")

        # ── TEST DELETE ───────────────────────────────────────────────────────
        if state.powers.get("delete", 0) > 0:
            # Find the smallest tile value present
            present = sorted({board[r][c] for r in range(4) for c in range(4) if board[r][c] > 0})
            if present:
                target = present[0]
                # Count how many we expect to be deleted
                count_before = sum(1 for r in range(4) for c in range(4) if board[r][c] == target)
                # Find one position to click
                pos = next((r, c) for r in range(4) for c in range(4) if board[r][c] == target)
                print(f"\n=== DELETE TEST ===")
                print(f"Deleting all {target}-tiles (found {count_before} of them, clicking ({pos[0]},{pos[1]}))")
                await execute_delete(page, pos[0], pos[1])
                state3 = await read_state(page)
                print("Board after delete:")
                print_board(state3)
                count_after = sum(1 for r in range(4) for c in range(4) if state3.board[r][c] == target)
                if count_after == 0:
                    print(f"  DELETE OK: all {count_before} × {target}-tiles removed")
                else:
                    print(f"  DELETE FAILED: {count_after} × {target}-tiles still present (had {count_before})")
            else:
                print("\n=== DELETE TEST SKIPPED (board is empty?) ===")
        else:
            print(f"\n=== DELETE TEST SKIPPED (no delete uses remaining — need 512 tile to unlock) ===")

        print("\nDone.")

    finally:
        await browser.close()
        await pw.stop()


if __name__ == "__main__":
    asyncio.run(main())
