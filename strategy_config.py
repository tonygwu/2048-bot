"""Centralized tunable knobs for strategy evaluation and depth policy."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DepthPolicy:
    min_depth: int = 5
    max_depth: int = 7
    soft_max: int = 7
    # Slight upward bias toward deeper search in open/mid-game.
    base: float = 1.45
    w_max: float = 0.25
    w_full: float = 1.10
    w_blocked: float = 0.75
    w_rough: float = 0.45
    rough_norm_divisor: float = 36.0
    open_empties_threshold: int = 8
    open_penalty: float = 0.40
    jammed_empties_threshold: int = 2
    # Offset baseline bump for jammed boards so late-game depth doesn't jump.
    jammed_bonus: float = 0.50
    low_valid_bonus_threshold: int = 2
    low_valid_bonus: float = 0.25
    # In open/mid-game boards, low legal-move count is often temporary noise.
    # Damp the low-valid bonus to avoid over-escalating depth too early.
    midgame_empties_threshold: int = 5
    midgame_rough_damp_threshold: float = 0.9
    midgame_low_valid_damp: float = 0.65
    surgery_empties_threshold: int = 3
    surgery_valid_threshold: int = 3
    surgery_rough_threshold: float = 0.9
    surgery_bonus: float = 0.40
    # Extra nudge for tight boards that usually need longer tactical search.
    tight_empties_threshold: int = 4
    tight_valid_threshold: int = 3
    tight_bonus: float = 0.50
    near_death_empties_threshold: int = 1
    # Depth-7 is now the hard ceiling; keep it reserved for true crises.
    near_death_valid_threshold: int = 2
    near_death_rough_threshold: float = 0.95
    near_death_max_log_threshold: float = 11.0


@dataclass(frozen=True)
class EvalWeights:
    empty: float = 270.0
    gradient: float = 2.2
    monotonicity: float = 47.0
    roughness: float = 35.0
    merge_potential: float = 65.0
    max_tile_log: float = 120.0
    sum_log: float = 140.0
    mobility: float = 95.0
    corner: float = 80.0
    powerup: float = 1.0
    promotion: float = 350.0
    high_merge: float = 180.0


@dataclass(frozen=True)
class PowerUpPolicy:
    w_base: float = 40.0
    delete_scale: float = 2.0
    max_swap_uses: int = 2
    max_delete_uses: int = 2
    swap_unlock_tile: int = 256
    delete_unlock_tile: int = 512
    prox_scale: float = 0.25
    prox_swap_min_tile: int = 128
    prox_delete_min_tile: int = 256
    # Power-up spend gate: require power-up actions to beat the best move by a
    # margin that adapts to board pressure (fullness + low mobility + roughness).
    spend_margin_calm: float = 130.0
    spend_margin_pressure: float = 30.0
    reserve_margin_per_extra_use: float = 30.0
    pressure_fullness_w: float = 0.45
    pressure_mobility_w: float = 0.45
    pressure_roughness_w: float = 0.10
    pressure_roughness_norm_divisor: float = 36.0
    swap_margin_mult: float = 1.35
    delete_margin_mult: float = 0.9
    # Swap search now scores all distinct non-empty tile pairs with the static
    # evaluator, then runs expectimax on a pressure-adaptive top-N shortlist.
    swap_pair_cap_calm: int = 20
    swap_pair_cap_pressure: int = 36
    # Undo triggers when post-action board quality drops sharply.
    undo_drop_trigger: float = 280.0
    # Also trigger when realized board quality misses planned board quality.
    undo_plan_gap_trigger: float = 190.0
    # Reuse the same board-pressure concept used by swap/delete spend gating.
    undo_pressure_relief: float = 90.0
    # Spend undo more freely when we are at full bank.
    undo_bank_uses_threshold: int = 2
    undo_bank_relief: float = 60.0
    # Spend undo more freely when the board is close to generating the next
    # unlock tile (e.g., 64+64 toward 128).
    undo_unlock_tile: int = 128
    # Start considering proximity once the run has reached at least this tile.
    undo_prox_min_tile: int = 64
    undo_prox_relief: float = 35.0
    # Keep triggers bounded so we do not overfire in calm boards.
    undo_trigger_floor: float = 70.0
    # Absolute drops can be noisy at high eval scales; require relative cliffs too.
    # Require a meaningful relative cliff before spending Undo.
    undo_drop_ratio_trigger: float = 0.08
    undo_plan_gap_ratio_trigger: float = 0.03
    # Plan-gap-only undo can overfire when spawn randomness shifts expected value
    # despite a materially improved realized board. Allow this much improvement
    # before suppressing a plan-gap-only undo trigger.
    undo_plan_gap_gain_tolerance: float = 120.0
    # Late-stage promotion: when max tile is very high but runner-up tile is still
    # behind, relax spend margins to allow tactical swap/delete intervention.
    late_stage_tile_threshold: int = 4096
    ultra_late_tile_threshold: int = 8192
    late_stage_second_tile_threshold: int = 4096
    late_stage_margin_mult: float = 0.65
    ultra_late_margin_mult: float = 0.40
    # When measuring whether undo paid off, look ahead this many actions.
    undo_success_horizon_actions: int = 4
    undo_success_margin: float = 120.0


DEFAULT_DEPTH_POLICY = DepthPolicy()
DEFAULT_EVAL_WEIGHTS = EvalWeights()
DEFAULT_POWERUP_POLICY = PowerUpPolicy()
