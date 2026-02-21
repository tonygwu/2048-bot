"""
game.py — Playwright interface for play2048.co

The game renders on an OffscreenCanvas in a Web Worker.
Board state is read via Worker postMessage interception (fast, exact)
with screenshot pixel-sampling as fallback.
"""

import asyncio
import io
from dataclasses import dataclass, field
from playwright.async_api import async_playwright, Page, BrowserContext

URL = "https://play2048.co/"
MOVE_DELAY = 0.05   # fallback delay only if worker-update wait times out
UPDATE_WAIT_TIMEOUT_MS = 220


@dataclass
class GameState:
    board: list[list[int]]   # 4x4 grid, 0 = empty
    score: int
    best: int
    over: bool
    won: bool
    powers: dict = field(default_factory=dict)  # {"undo": N, "swap": N, "delete": N}


ARROW_KEYS = {
    "up":    "ArrowUp",
    "down":  "ArrowDown",
    "left":  "ArrowLeft",
    "right": "ArrowRight",
}

# ── Worker postMessage interceptor ───────────────────────────────────────────
# Injected before any page JS runs. Captures all messages to/from Web Workers.
WORKER_INTERCEPT_JS = """
(function() {
    window._workerMsgs = [];
    window._lastWorkerUpdate = null;
    window._workerUpdateSeq = 0;

    const OrigWorker = window.Worker;
    window.Worker = function(url, options) {
        const worker = new OrigWorker(url, options);

        const origPost = worker.postMessage.bind(worker);
        worker.postMessage = function(data, transferOrOptions) {
            try {
                window._workerMsgs.push({ dir: 'to', data: JSON.parse(JSON.stringify(data)) });
                if (data && data.type === 'call' && data.call === 'update') {
                    window._lastWorkerUpdate = JSON.parse(JSON.stringify(data));
                    window._workerUpdateSeq += 1;
                }
                if (window._workerMsgs.length > 200) window._workerMsgs.splice(0, 100);
            } catch(e) {}
            return transferOrOptions !== undefined
                ? origPost(data, transferOrOptions)
                : origPost(data);
        };

        worker.addEventListener('message', function(e) {
            try {
                window._workerMsgs.push({ dir: 'from', data: JSON.parse(JSON.stringify(e.data)) });
                if (e.data && e.data.type === 'call' && e.data.call === 'update') {
                    window._lastWorkerUpdate = JSON.parse(JSON.stringify(e.data));
                    window._workerUpdateSeq += 1;
                }
                if (window._workerMsgs.length > 200) window._workerMsgs.splice(0, 100);
            } catch(e) {}
        });

        return worker;
    };
    Object.setPrototypeOf(window.Worker, OrigWorker);
    window.Worker.prototype = OrigWorker.prototype;
})();
"""


async def launch_browser(headless: bool = False):
    """Start Playwright, inject interceptor, navigate. Returns (pw, browser, page)."""
    pw = await async_playwright().start()
    browser = await pw.chromium.launch(headless=headless)
    context: BrowserContext = await browser.new_context()
    await context.add_init_script(WORKER_INTERCEPT_JS)
    page = await context.new_page()
    await page.goto(URL, wait_until="domcontentloaded", timeout=60000)
    await page.wait_for_selector("canvas", timeout=15000)
    await asyncio.sleep(0.5)
    # Close welcome overlay with Escape
    await page.keyboard.press("Escape")
    await asyncio.sleep(0.2)
    # Wait until the first worker "update" message arrives (initial board state).
    try:
        await page.wait_for_function(
            "() => (window._workerUpdateSeq || 0) > 0",
            timeout=6000,
        )
    except Exception:
        # Fallback to eventual read_state screenshot path if worker stream is delayed.
        pass
    return pw, browser, page


# ── Score reading (from DOM) ──────────────────────────────────────────────────
async def read_score(page: Page) -> tuple[int, int]:
    """Return (score, best) by reading the DOM score spans."""
    # From diagnostic: score and best are in span.shrink-1.truncate elements
    # The header has: "Score" label, score value, "Best" label, best value
    spans = await page.locator("span.shrink-1.truncate").all_inner_texts()
    # spans should be ["<score>", "<best>"] or similar
    def safe_int(s: str) -> int:
        try:
            return int(s.strip().replace(",", ""))
        except Exception:
            return 0
    if len(spans) >= 2:
        return safe_int(spans[0]), safe_int(spans[1])
    return 0, 0


# ── Board reading (via Worker messages) ───────────────────────────────────────
def _parse_update_board(data: dict) -> list[list[int]] | None:
    """
    Parse a worker 'update' message.
    Format: {"type":"call","call":"update","args":[{"state":"...","board":[[...],...],...}]}
    board[y][x] is null (empty) or {"value": N, "position": {"x": col, "y": row}, ...}
    Returns a 4x4 int grid or None.
    """
    if not isinstance(data, dict):
        return None
    if data.get("type") != "call" or data.get("call") != "update":
        return None
    args = data.get("args")
    if not isinstance(args, list) or len(args) == 0:
        return None
    state_obj = args[0]
    if not isinstance(state_obj, dict):
        return None
    raw_board = state_obj.get("board")
    if not isinstance(raw_board, list) or len(raw_board) != 4:
        return None

    board = [[0] * 4 for _ in range(4)]
    for row_cells in raw_board:
        if not isinstance(row_cells, list):
            return None
        for cell in row_cells:
            if cell is None:
                continue
            if not isinstance(cell, dict):
                continue
            val = cell.get("value")
            pos = cell.get("position")
            if val is None or pos is None:
                continue
            col = pos.get("x")
            row = pos.get("y")
            if col is None or row is None:
                continue
            board[row][col] = int(val)
    return board


def _parse_powerups(data: dict) -> dict:
    """Extract power-up use counts from an 'update' worker message.
    Returns {"undo": N, "swap": N, "delete": N} (all 0 on failure)."""
    zero = {"undo": 0, "swap": 0, "delete": 0}
    try:
        args = data.get("args", [])
        if not args or not isinstance(args[0], dict):
            return zero
        pu = args[0].get("powerups", {})
        if not isinstance(pu, dict):
            return zero
        return {
            "undo":   int(pu.get("undo",                 {}).get("usesRemaining", 0)),
            "swap":   int(pu.get("swapTwoTiles",         {}).get("usesRemaining", 0)),
            "delete": int(pu.get("removeTilesByValue",   {}).get("usesRemaining", 0)),
        }
    except Exception:
        return zero


async def read_board_from_worker(page: Page) -> list[list[int]] | None:
    """Read the board from the most recent 'update' worker message."""
    msgs = await page.evaluate("() => window._workerMsgs || []")
    for msg in reversed(msgs):
        board = _parse_update_board(msg.get("data"))
        if board is not None:
            return board
    return None


# ── Screenshot pixel fallback ─────────────────────────────────────────────────
# Cell centers in the 576-unit canvas coordinate space.
# From SVG: cells at x/y = 54,174,294,414 with size 108 → centers at 108,228,348,468.
_CELL_NORM = [108/576, 228/576, 348/576, 468/576]

# Color → tile value map. Built empirically; empty cell ≈ (189,172,151).
# Populated by calibration run; unknown colors are treated as 0.
_COLOR_MAP: dict[tuple[int,int,int], int] = {}
_EMPTY_COLOR = (189, 172, 151)
_COLOR_TOLERANCE = 20


def _color_distance(a: tuple, b: tuple) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def _lookup_color(rgb: tuple) -> int:
    """Map an RGB sample to a tile value. 0 = empty/unknown."""
    best_dist, best_val = float("inf"), 0
    for known_rgb, val in _COLOR_MAP.items():
        d = _color_distance(rgb, known_rgb)
        if d < best_dist:
            best_dist, best_val = d, val
    if best_dist < _COLOR_TOLERANCE:
        return best_val
    # Treat as empty if close to empty color
    if _color_distance(rgb, _EMPTY_COLOR) < _COLOR_TOLERANCE * 2:
        return 0
    return -1   # unknown non-empty tile


async def read_board_from_screenshot(page: Page) -> list[list[int]] | None:
    """Fallback: screenshot the canvas and sample cell center pixels."""
    from PIL import Image
    canvas_el = page.locator("canvas").first
    box = await canvas_el.bounding_box()
    if not box:
        return None
    png = await page.screenshot(clip=box)
    img = Image.open(io.BytesIO(png)).convert("RGB")
    W, H = img.size
    board = []
    for row in range(4):
        board_row = []
        for col in range(4):
            px = int(_CELL_NORM[col] * W)
            py = int(_CELL_NORM[row] * H)
            rgb = img.getpixel((px, py))
            board_row.append(_lookup_color(rgb))
        board.append(board_row)
    return board


# ── Main state reader ─────────────────────────────────────────────────────────
async def read_state(page: Page) -> GameState:
    # Single browser round-trip for worker snapshot + score + overlay flags.
    snap = await page.evaluate(
        """
        () => {
          const update = window._lastWorkerUpdate || null;
          const spans = Array.from(document.querySelectorAll('span.shrink-1.truncate'));
          const scoreText = spans[0] ? (spans[0].textContent || '') : '';
          const bestText = spans[1] ? (spans[1].textContent || '') : '';
          const bodyText = document.body ? (document.body.innerText || '') : '';
          return {
            update,
            scoreText,
            bestText,
            over: bodyText.includes('Game over'),
            won: bodyText.includes('You win'),
          };
        }
        """
    )

    def _safe_int(s: str) -> int:
        try:
            return int((s or "").strip().replace(",", ""))
        except Exception:
            return 0

    score = _safe_int(snap.get("scoreText", ""))
    best = _safe_int(snap.get("bestText", ""))
    board = _parse_update_board(snap.get("update"))
    powers: dict = _parse_powerups(snap.get("update") or {})
    if board is None:
        board = await read_board_from_screenshot(page) or [[0]*4 for _ in range(4)]
    over = bool(snap.get("over"))
    won = bool(snap.get("won"))
    return GameState(board=board, score=score, best=best, over=over, won=won, powers=powers)


# ── Move execution ────────────────────────────────────────────────────────────
async def execute_move(page: Page, direction: str) -> None:
    seq = await page.evaluate("() => window._workerUpdateSeq || 0")
    # Retry once with explicit canvas focus if the first keypress is dropped.
    for attempt in range(2):
        if attempt > 0:
            try:
                await page.locator("canvas").first.click(timeout=300)
            except Exception:
                pass
        await page.keyboard.press(ARROW_KEYS[direction])
        try:
            await page.wait_for_function(
                "prev => (window._workerUpdateSeq || 0) > prev",
                arg=seq,
                timeout=UPDATE_WAIT_TIMEOUT_MS,
            )
            return
        except Exception:
            pass
    await asyncio.sleep(MOVE_DELAY)


# ── Power-up execution ────────────────────────────────────────────────────────
POWER_DELAY = 0.35   # seconds to wait after each power-up interaction step


async def _wait_for_worker_update(page: Page, prev_seq: int, fallback_delay: float = MOVE_DELAY) -> None:
    try:
        await page.wait_for_function(
            "prev => (window._workerUpdateSeq || 0) > prev",
            arg=prev_seq,
            timeout=UPDATE_WAIT_TIMEOUT_MS,
        )
    except Exception:
        await asyncio.sleep(fallback_delay)


async def _power_buttons(page: Page) -> list:
    """Return the three power-up buttons (those with 'aspect-square' in their class)."""
    return await page.query_selector_all("button[class*='aspect-square']")


async def click_tile(page: Page, row: int, col: int) -> None:
    """Click on the canvas tile at grid position (row, col)."""
    box = await page.locator("canvas").first.bounding_box()
    if not box:
        return
    x = box["x"] + _CELL_NORM[col] * box["width"]
    y = box["y"] + _CELL_NORM[row] * box["height"]
    seq = await page.evaluate("() => window._workerUpdateSeq || 0")
    await page.mouse.click(x, y)
    await _wait_for_worker_update(page, seq, fallback_delay=POWER_DELAY)


async def execute_undo(page: Page) -> None:
    """Use the Undo power-up (first aspect-square button)."""
    btns = await _power_buttons(page)
    if btns:
        seq = await page.evaluate("() => window._workerUpdateSeq || 0")
        await btns[0].click()
        await _wait_for_worker_update(page, seq, fallback_delay=POWER_DELAY)


async def execute_undo_on_gameover(page: Page) -> None:
    """Click the Undo button on the game-over overlay to resume play from the previous state."""
    btns = page.locator("button:has-text('Undo')")
    # The game-over overlay button can appear a moment after the final move.
    # Wait briefly, then click a visible "Undo" button if present.
    try:
        await btns.first.wait_for(state="visible", timeout=2500)
    except Exception:
        pass

    target = None
    try:
        count = await btns.count()
    except Exception:
        count = 0
    for i in range(count):
        cand = btns.nth(i)
        try:
            if await cand.is_visible():
                target = cand
                break
        except Exception:
            continue

    if target is not None:
        try:
            await target.click(timeout=3000)
        except Exception:
            try:
                await target.click(force=True, timeout=2000)
            except Exception:
                pass
        await asyncio.sleep(POWER_DELAY)
        return

    # Fallback: try the regular header power-up Undo if no overlay button found.
    await execute_undo(page)


async def execute_swap(page: Page, r1: int, c1: int, r2: int, c2: int) -> None:
    """Use the Swap power-up: activate, then click the two tile positions."""
    btns = await _power_buttons(page)
    if len(btns) >= 2:
        seq = await page.evaluate("() => window._workerUpdateSeq || 0")
        await btns[1].click()
        await _wait_for_worker_update(page, seq, fallback_delay=POWER_DELAY)
        await click_tile(page, r1, c1)
        await click_tile(page, r2, c2)


async def execute_delete(page: Page, row: int, col: int) -> None:
    """Use the Delete power-up: activate, then click a tile of the target value."""
    btns = await _power_buttons(page)
    if len(btns) >= 3:
        seq = await page.evaluate("() => window._workerUpdateSeq || 0")
        await btns[2].click()
        await _wait_for_worker_update(page, seq, fallback_delay=POWER_DELAY)
        await click_tile(page, row, col)


async def dismiss_win_overlay(page: Page) -> None:
    """Dismiss the 'You win!' overlay so the game can continue past 2048."""
    for txt in ["Keep going!", "Keep going", "Keep Going!", "Keep Going", "Continue"]:
        btn = page.locator(f"text={txt}")
        if await btn.count() > 0:
            try:
                await btn.first.click(timeout=3000)
                await asyncio.sleep(0.3)
                return
            except Exception:
                pass
    await page.keyboard.press("Escape")
    await asyncio.sleep(0.3)


async def new_game(page: Page) -> None:
    """Start a new game.

    After game-over an overlay may cover the header buttons.
    Try clicking visible 'New Game' buttons; fall back to Escape + retry.
    """
    # First try to find a visible New Game button (may be in game-over overlay)
    for selector in ["text=New Game", "text=Try again", "text=Play again"]:
        btn = page.locator(selector)
        count = await btn.count()
        for i in range(count):
            try:
                await btn.nth(i).click(timeout=3000)
                await asyncio.sleep(0.4)
                return
            except Exception:
                pass

    # Dismiss any overlay with Escape, then try header button with force
    await page.keyboard.press("Escape")
    await asyncio.sleep(0.3)
    btn = page.locator("text=New Game")
    if await btn.count() > 0:
        await btn.first.click(force=True)
    await asyncio.sleep(0.4)


# ── Display ───────────────────────────────────────────────────────────────────
def print_board(state: GameState) -> None:
    print(f"Score: {state.score}  Best: {state.best}  over={state.over}")
    print("+------+------+------+------+")
    for row in state.board:
        cells = "".join(f"{v:^6}" if v else "      " for v in row)
        print(f"|{cells[0:6]}|{cells[6:12]}|{cells[12:18]}|{cells[18:24]}|")
        print("+------+------+------+------+")


# ── Diagnostic: explore worker messages ──────────────────────────────────────
async def _dump_worker_msgs(page: Page, n: int = 20) -> None:
    import json
    msgs = await page.evaluate("() => window._workerMsgs || []")
    print(f"\n{len(msgs)} worker messages captured. Showing last {n}:")
    for msg in msgs[-n:]:
        direction = msg.get("dir", "?")
        data_str = json.dumps(msg.get("data"), default=str)[:300]
        print(f"  [{direction}] {data_str}")


# ── Smoke test ────────────────────────────────────────────────────────────────
async def _smoke_test():
    print("Launching browser… (headless=False so you can watch)")
    pw, browser, page = await launch_browser(headless=False)
    try:
        state = await read_state(page)
        print("\nInitial board:")
        print_board(state)

        for direction in ["up", "right", "down", "left", "up", "left"]:
            await execute_move(page, direction)
            state = await read_state(page)
            print(f"\nAfter move '{direction}':")
            print_board(state)
            if state.over:
                print("Game over!")
                break

        print("\nSmoke test passed. Closing in 5s…")
        await asyncio.sleep(5)
    finally:
        await browser.close()
        await pw.stop()


if __name__ == "__main__":
    asyncio.run(_smoke_test())
