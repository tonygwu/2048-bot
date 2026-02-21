"""Centralized tunable knobs for strategy evaluation and depth policy."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DepthPolicy:
    min_depth: int = 3
    max_depth: int = 6
    soft_max: int = 6
    # Slight upward bias toward deeper search in open/mid-game.
    base: float = 0.95
    w_max: float = 0.22
    w_full: float = 1.10
    w_blocked: float = 0.75
    w_rough: float = 0.45
    rough_norm_divisor: float = 36.0
    open_empties_threshold: int = 8
    open_penalty: float = 0.70
    jammed_empties_threshold: int = 2
    # Offset baseline bump for jammed boards so late-game depth doesn't jump.
    jammed_bonus: float = 0.23
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
    tight_bonus: float = 0.35
    near_death_empties_threshold: int = 1
    # Depth-6 should remain rare, but allow two-move late-game traps.
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


DEFAULT_DEPTH_POLICY = DepthPolicy()
DEFAULT_EVAL_WEIGHTS = EvalWeights()
DEFAULT_POWERUP_POLICY = PowerUpPolicy()
