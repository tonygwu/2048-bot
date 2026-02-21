"""Compatibility facade for strategy modules.

This module preserves the historical `from strategy import ...` API while the
implementation is split across focused modules:
- strategy_actions.py
- strategy_core.py
- strategy_eval.py
- strategy_search.py
"""

from strategy_actions import (  # noqa: F401
    Action,
    ActionTuple,
    DeleteAction,
    MoveAction,
    SwapAction,
    action_from_tuple,
    action_to_tuple,
)
from strategy_config import (  # noqa: F401
    DEFAULT_DEPTH_POLICY,
    DEFAULT_EVAL_WEIGHTS,
    DEFAULT_POWERUP_POLICY,
    DepthPolicy,
    EvalWeights,
    PowerUpPolicy,
)
from strategy_core import DIRECTIONS, apply_move, board_to_bb, empty_cells, is_game_over  # noqa: F401
from strategy_eval import (  # noqa: F401
    SCORE_BOARD_VERSION,
    EvalFeatures,
    _monotonicity,
    _roughness,
    drain_new_entries,
    extract_eval_features,
    get_trans_stats,
    load_trans_table,
    reset_trans_stats,
    score_board,
    score_from_features,
)
from strategy_search import (  # noqa: F401
    _expectimax,
    apply_delete,
    apply_swap,
    auto_depth,
    best_action,
    best_action_obj,
    best_move,
)

__all__ = [
    "Action",
    "ActionTuple",
    "DEFAULT_DEPTH_POLICY",
    "DEFAULT_EVAL_WEIGHTS",
    "DEFAULT_POWERUP_POLICY",
    "DIRECTIONS",
    "DeleteAction",
    "DepthPolicy",
    "EvalFeatures",
    "EvalWeights",
    "MoveAction",
    "PowerUpPolicy",
    "SCORE_BOARD_VERSION",
    "SwapAction",
    "_expectimax",
    "_monotonicity",
    "_roughness",
    "action_from_tuple",
    "action_to_tuple",
    "apply_delete",
    "apply_move",
    "apply_swap",
    "auto_depth",
    "best_action",
    "best_action_obj",
    "best_move",
    "board_to_bb",
    "drain_new_entries",
    "empty_cells",
    "extract_eval_features",
    "get_trans_stats",
    "is_game_over",
    "load_trans_table",
    "reset_trans_stats",
    "score_board",
    "score_from_features",
]
