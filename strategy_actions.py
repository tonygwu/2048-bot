"""Action models and compatibility conversion helpers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MoveAction:
    direction: str


@dataclass(frozen=True)
class SwapAction:
    r1: int
    c1: int
    r2: int
    c2: int


@dataclass(frozen=True)
class DeleteAction:
    value: int
    row: int
    col: int


Action = MoveAction | SwapAction | DeleteAction
ActionTuple = tuple


def action_to_tuple(action: Action | None) -> ActionTuple | None:
    """Convert typed action objects to the legacy tuple action format."""
    if action is None:
        return None
    if isinstance(action, MoveAction):
        return ("move", action.direction)
    if isinstance(action, SwapAction):
        return ("swap", action.r1, action.c1, action.r2, action.c2)
    if isinstance(action, DeleteAction):
        return ("delete", action.value, action.row, action.col)
    raise TypeError(f"Unknown action type: {type(action)!r}")


def action_from_tuple(action: ActionTuple | None) -> Action | None:
    """Convert legacy tuple actions to typed action objects."""
    if action is None:
        return None
    kind = action[0]
    if kind == "move":
        return MoveAction(direction=action[1])
    if kind == "swap":
        _, r1, c1, r2, c2 = action
        return SwapAction(r1=r1, c1=c1, r2=r2, c2=c2)
    if kind == "delete":
        _, value, row, col = action
        return DeleteAction(value=value, row=row, col=col)
    raise ValueError(f"Unknown action tuple kind: {kind!r}")
