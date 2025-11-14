"""
hivenet_td.py

Value-only Hive network + batched successor evaluation and move selection.

Usage:
  from hivegame.hivenet_td import HiveValueNet, make_value_model
  model = make_value_model()
  cmd = model.select_move_by_value(hive, color, turn)
"""
from typing import Optional, List, Tuple, Any
import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# Use existing helpers from your project
from hivegame.hivenet_train import (
    encode_board,
    _current_board_state_from_hive,
    _reserves_from_hive,
    _apply_move_on_hive,
    IN_CHANNELS,
    GRID_SIZE,
    RESERVE_VEC_LEN,
)


# -------------------- move normalization helpers --------------------

def _direction_to_poc_str(hive, direction: Any) -> Optional[str]:
    """Map an engine direction (int or engine const) to CLI point-of-contact token."""
    # Prefer engine constants if available
    dir_map = {
        getattr(hive, "W", None): "|*",
        getattr(hive, "NW", None): "/*",
        getattr(hive, "NE", None): "*\\",
        getattr(hive, "E", None):  "*|",
        getattr(hive, "SE", None): "*/",
        getattr(hive, "SW", None): "\\*",
        getattr(hive, "O", None):  "=*",
    }
    # exact match of engine const
    for k, token in dir_map.items():
        if k is not None and direction == k:
            return token

    # integer directions (robust guess: 0..6 -> [W, NW, NE, E, SE, SW, O])
    if isinstance(direction, int):
        order = ["|*", "/*", "*\\", "*|", "*/", "\\*", "=*"]
        if 0 <= direction < len(order):
            return order[direction]

    # string already a poc?
    if isinstance(direction, str) and direction in {"|*","/*","*\\","*|","*/","\\*","=*"}:
        return direction

    return None


def _move_to_cmd_str(hive, move: Any, turn: int) -> Optional[str]:
    """
    Convert a move (string or various structured formats) into CLI string:
       - 'pass'
       - 'wA1' (first placement)
       - 'bS1<POC>wA1' for others
    Return None if cannot convert.
    """
    # already a CLI string
    if isinstance(move, str):
        m = move.strip()
        if m == "pass" or len(m) == 3 or len(m) == 8:
            return m
        # If it's something else, give it a chance anyway; exec_cmd will validate.
        return m

    # tuple/object formats used by engine/agents:
    # 1) ('non_play','pass')
    if isinstance(move, (tuple, list)) and len(move) == 2 and move[0] == 'non_play':
        return "pass"

    # 2) ('play' or 'move', (actPiece, refPiece, direction))
    if isinstance(move, (tuple, list)) and len(move) == 2 and move[0] in ('play', 'move'):
        inner = move[1]
        if isinstance(inner, (tuple, list)) and len(inner) == 3:
            actPiece, refPiece, direction = inner
            # if first placement (no ref/direction) AND it's the first turn, 3-char is valid
            if (refPiece is None or refPiece == "") and (direction is None) and turn <= 1:
                if isinstance(actPiece, str) and len(actPiece) == 3:
                    return actPiece
            # otherwise need a POC token + refPiece
            poc = _direction_to_poc_str(hive, direction)
            if isinstance(actPiece, str) and isinstance(refPiece, str) and poc is not None:
                return f"{actPiece}{poc}{refPiece}"
            # fallback: cannot normalize
            return None

    # 3) a 3-length tuple like ('w','A','1') -> try to stringify
    if isinstance(move, (tuple, list)) and len(move) == 3 and all(isinstance(x, str) for x in move):
        candidate = "".join(move)
        if len(candidate) == 3:
            return candidate

    return None


def _apply_move_on_clone_flexible(hive, move: Any, turn: int) -> bool:
    """Apply a move (flexibly parsed) onto a deep-copied hive and return True if accepted."""
    hive_clone = copy.deepcopy(hive)

    # Prefer the engine's own action API if we can parse the structured form;
    # otherwise, normalize to CLI string and use _apply_move_on_hive.
    cmd = _move_to_cmd_str(hive_clone, move, turn)
    if cmd is None:
        # try raw on engine if it's already structured in the engine's shape
        try:
            if isinstance(move, (tuple, list)) and len(move) == 2:
                if move[0] == 'non_play' and move[1] == 'pass':
                    hive_clone.action('non_play', 'pass')
                    return True
                if move[0] in ('play', 'move'):
                    actPiece, refPiece, direction = move[1]
                    hive_clone.action('play', (actPiece, refPiece, direction))
                    return True
        except Exception:
            return False
        return False

    try:
        _apply_move_on_hive(hive_clone, cmd, turn)
        return True
    except Exception:
        return False


class HiveValueNet(nn.Module):
    """
    Value-only network for Hive. Input: (state_tensor, reserve_vec) -> scalar in [-1,1].
    The architecture mirrors the earlier backbone and value head from your codebase.
    """

    def __init__(
        self,
        in_channels: int,
        board_size: Tuple[int, int],
        reserve_vec_len: int = RESERVE_VEC_LEN,
        value_hidden: int = 128,
    ):
        super().__init__()
        self.board_size = board_size

        # Shared convolutional backbone (same layout as your dual-headed net)
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Value head
        self.value_conv = nn.Conv2d(128, 8, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(8)
        flat_size = 8 * board_size[0] * board_size[1]
        self.value_fc1 = nn.Linear(flat_size + reserve_vec_len, value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, 1)

    def forward(self, x: torch.Tensor, reserve_vec: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
         - x: (B, C, H, W)
         - reserve_vec: (B, R)
        Returns:
         - value: (B,) tensor in [-1,1]
        """
        features = self.backbone(x)
        v = F.relu(self.value_bn(self.value_conv(features)))
        v = v.view(v.size(0), -1)
        v = torch.cat([v, reserve_vec], dim=1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v)).squeeze(-1)
        return value

    # -------------------- utilities for move selection --------------------

    def evaluate_successors(
        self,
        successor_states: torch.Tensor,
        successor_reserves: torch.Tensor,
        device: Optional[torch.device] = None,
        batch_size: Optional[int] = None,
    ) -> List[float]:
        """
        Evaluate many successor states in batches.
          - successor_states: (N, C, H, W) (torch.FloatTensor)
          - successor_reserves: (N, R)
        Returns a Python list of N floats (values).
        """
        device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.to(device)
        self.eval()

        N = successor_states.shape[0]
        batch_size = batch_size or min(N, 256)

        values = []
        with torch.no_grad():
            for i in range(0, N, batch_size):
                xb = successor_states[i : i + batch_size].to(device)
                rb = successor_reserves[i : i + batch_size].to(device)
                preds = self(xb, rb)
                preds = preds.detach().cpu()
                values.append(preds)
        values = torch.cat(values, dim=0)
        return values.tolist()

    def select_move_by_value(
        self,
        hive,
        color: str,
        turn: int,
        device: Optional[torch.device] = None,
        batch_size: Optional[int] = None,
        tie_break: str = "random",
        explore_eps: float = 0.0,
    ) -> str:
        """
        Choose a legal move by evaluating all successors with the value net.
        - Uses hive.find_valid_moves(color) as the authoritative legal list.
        - If explore_eps > 0, performs epsilon-greedy exploration.
        Returns a CLI move string (e.g., 'wA1', 'Q01A02', or 'pass').
        """
        # 1) get legal moves
        try:
            legal_moves = list(hive.find_valid_moves(color))
        except Exception as e:
            raise RuntimeError("Engine must provide find_valid_moves(color) for select_move_by_value") from e

        if not legal_moves:
            return "pass"

        # epsilon-greedy exploration
        if explore_eps > 0.0 and random.random() < explore_eps:
            mv = random.choice(legal_moves)
            cmd = _move_to_cmd_str(hive, mv, turn)
            return cmd if cmd is not None else "pass"

        # Build successor encodings
        succ_states = []
        succ_reserves = []
        succ_moves = []

        for mv in legal_moves:
            try:
                h_clone = copy.deepcopy(hive)
                _apply_move_on_clone_flexible(h_clone, mv, turn)
            except Exception:
                continue

            next_color = "w" if color == "b" else "b"
            bs = _current_board_state_from_hive(h_clone)
            rs = _reserves_from_hive(h_clone)
            st_tensor, res_vec = encode_board(bs, next_color, turn + 1, reserves=rs)

            succ_states.append(st_tensor)
            succ_reserves.append(res_vec)
            succ_moves.append(mv)

        if not succ_moves:
            return "pass"

        # Stack into batched tensors
        X_states = torch.stack(succ_states, dim=0)  # (N,C,H,W)
        X_res = torch.stack(succ_reserves, dim=0)  # (N,R)
        values = self.evaluate_successors(X_states, X_res, device=device, batch_size=batch_size)

        # pick best
        max_v = max(values)
        best_indices = [i for i, v in enumerate(values) if v == max_v]
        if tie_break == "random" and len(best_indices) > 1:
            choice_idx = random.choice(best_indices)
        else:
            choice_idx = best_indices[0]

        # normalize returned move to CLI string
        chosen = succ_moves[choice_idx]
        cmd = _move_to_cmd_str(hive, chosen, turn)
        return cmd if cmd is not None else "pass"

    def select_move_by_value_subset(
        self,
        hive,
        color: str,
        turn: int,
        candidate_k: int = 12,
        device: Optional[torch.device] = None,
        batch_size: Optional[int] = None,
        tie_break: str = "random",
        explore_eps: float = 0.0,
    ) -> str:
        """
        Choose a legal move by evaluating a random subset (candidate_k) of successors with the value net.
        - Uses hive.find_valid_moves(color) as the authoritative legal list.
        - If explore_eps > 0, performs epsilon-greedy exploration.
        Returns a CLI move string (e.g., 'wA1', 'Q01A02', or 'pass').
        """
        try:
            legal_moves = list(hive.find_valid_moves(color))
        except Exception as e:
            raise RuntimeError("Engine must provide find_valid_moves(color) for select_move_by_value_subset") from e

        if not legal_moves:
            return "pass"

        # epsilon-greedy exploration
        if explore_eps > 0.0 and random.random() < explore_eps:
            mv = random.choice(legal_moves)
            cmd = _move_to_cmd_str(hive, mv, turn)
            return cmd if cmd is not None else "pass"

        # Candidate sampling
        k = min(candidate_k, len(legal_moves))
        candidates = random.sample(legal_moves, k) if len(legal_moves) > k else list(legal_moves)

        succ_states = []
        succ_reserves = []
        succ_moves = []

        for mv in candidates:
            try:
                h_clone = copy.deepcopy(hive)
                _apply_move_on_clone_flexible(h_clone, mv, turn)
            except Exception:
                continue

            next_color = "w" if color == "b" else "b"
            bs = _current_board_state_from_hive(h_clone)
            rs = _reserves_from_hive(h_clone)
            st_tensor, res_vec = encode_board(bs, next_color, turn + 1, reserves=rs)

            succ_states.append(st_tensor)
            succ_reserves.append(res_vec)
            succ_moves.append(mv)

        if not succ_moves:
            return "pass"

        X_states = torch.stack(succ_states, dim=0)
        X_res = torch.stack(succ_reserves, dim=0)
        values = self.evaluate_successors(X_states, X_res, device=device, batch_size=batch_size)

        max_v = max(values)
        best_indices = [i for i, v in enumerate(values) if v == max_v]
        if tie_break == "random" and len(best_indices) > 1:
            choice_idx = random.choice(best_indices)
        else:
            choice_idx = best_indices[0]

        chosen = succ_moves[choice_idx]
        cmd = _move_to_cmd_str(hive, chosen, turn)
        return cmd if cmd is not None else "pass"


def make_value_model(device: Optional[torch.device] = None) -> HiveValueNet:
    """Convenience constructor using your project's constants."""
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = HiveValueNet(in_channels=IN_CHANNELS, board_size=(GRID_SIZE, GRID_SIZE), reserve_vec_len=RESERVE_VEC_LEN)
    return model
