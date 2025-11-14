"""
MCTSWithValue: Monte Carlo Tree Search for Hive using a value network (no policy head).

- PUCT/UCT selection with uniform priors (optionally Dirichlet noise at root).
- Candidate move sampling at expansion.
- Batch value evaluation.
- No transpositions; tree is a dict keyed by state_key (tuple of move history).
- Only CLI move strings are used as edges (via _move_to_cmd_str).
"""

import time
import random
import copy
import math
import os
import json
from typing import Dict, Any, Optional, List, Tuple

import torch

from hivegame.hive import Hive
from hivegame.piece import HivePiece
from hivegame.hivenet_train import (
    HiveNet,
    encode_board,
    _current_board_state_from_hive,
    _reserves_from_hive,
    IN_CHANNELS,
    GRID_SIZE,
    MOVE_SPACE_SIZE,
    RESERVE_VEC_LEN,
)

# ---------------------------------------------------------------------
# Move vocabulary loader (so policy head aligns with your training vocab)
# ---------------------------------------------------------------------

class MoveVocab:
    def __init__(self, vocab_path: Optional[str] = None):
        # Default: try to load project-level move_vocab.json if it exists
        if vocab_path is None:
            # repo layout: .../hivegame/ -> sibling dir "hive" has move_vocab.json
            here = os.path.dirname(__file__)
            vocab_path = os.path.join(here, "..", "hive", "move_vocab.json")
        self.index_to_move: Dict[int, str] = {}
        self.move_to_index: Dict[str, int] = {}
        if os.path.isfile(vocab_path):
            with open(vocab_path, "r") as f:
                data = json.load(f)
            # accept either {"index_to_move": [...]} or {"move_to_index": {...}}
            if "index_to_move" in data:
                self.index_to_move = {int(k): v for k, v in data["index_to_move"].items()}
                self.move_to_index = {v: int(k) for k, v in self.index_to_move.items()}
            elif "move_to_index" in data:
                self.move_to_index = {k: int(v) for k, v in data["move_to_index"].items()}
                self.index_to_move = {v: k for k, v in self.move_to_index.items()}
        else:
            # Fallback: empty vocab -> no policy prior; MCTS will use uniform priors.
            self.index_to_move = {}
            self.move_to_index = {}

    def to_index(self, move_str: str) -> Optional[int]:
        return self.move_to_index.get(move_str)

    def to_move(self, index: int) -> Optional[str]:
        return self.index_to_move.get(index)

# ------------------------------------------------------
# Minimal adapters to the Hive engine (legal/apply/clone)
# ------------------------------------------------------

def active_player_and_color(hive) -> Tuple[str, str]:
    """
    Return a tuple (color_id, color_str) matching downstream expectations.
    Hive.get_active_player() already gives 'w' or 'b'.
    """
    color = hive.get_active_player()
    return color, color

def list_legal_moves(hive, color: str) -> List[Tuple]:
    """
    Directly use Hive.find_valid_moves(color).
    """
    try:
        moves = hive.find_valid_moves(color)
        return moves if moves else []
    except Exception:
        return [('non_play', 'pass')]

def get_board_state_and_reserves(hive) -> Tuple[List[Tuple[int, int, List[str]]], Dict[Tuple[str, str], int]]:
    """
    Use the already implemented helpers from hivenet_train.
    board_state: list of (q, r, stack_list_of_piece_names)
    reserves: {('w','Q'): count, ...}
    """
    board_state = _current_board_state_from_hive(hive)
    reserves = _reserves_from_hive(hive)
    return board_state, reserves

# ------------------------------------------------------
# Encode the current Hive state for the network
# ------------------------------------------------------

def is_terminal(hive) -> Tuple[bool, float]:
    """
    Terminal check with value from current side-to-move perspective.
    """
    result = hive.check_victory()
    if result == hive.UNFINISHED:
        return False, 0.0
    color = hive.get_active_player()
    if result == hive.WHITE_WIN:
        return True, 1.0 if color == 'w' else -1.0
    if result == hive.BLACK_WIN:
        return True, 1.0 if color == 'b' else -1.0
    if result == hive.DRAW:
        return True, 0.0
    return True, 0.0

# ------------------------------------------------------
# Move string <-> tuple helpers for policy mapping
# ------------------------------------------------------

def direction_to_poc(direction, hive) -> str:
    # Map internal direction constants to Hive notation POC
    dir_map = {
        getattr(hive, "W", None): '|*',
        getattr(hive, "NW", None): '/*',
        getattr(hive, "NE", None): '*\\',
        getattr(hive, "E", None): '*|',
        getattr(hive, "SE", None): '*/',
        getattr(hive, "SW", None): '\\*',
        getattr(hive, "O", None): '=*',
    }
    return dir_map.get(direction, '')

def move_tuple_to_string(move, hive) -> str:
    """
    Convert ('play'/'move', (act, ref, dir)) to Hive notation string used by your vocab.
    """
    if move[0] == 'non_play' or move == 'pass':
        return 'pass'
    kind, (actPiece, refPiece, direction) = move
    if refPiece is None or direction is None:
        return actPiece  # placement with no ref (opening)
    poc = direction_to_poc(direction, hive)
    return f"{actPiece}{poc}{refPiece}"

# ------------------------------------------------------
# Network evaluator
# ------------------------------------------------------

class TDPolicyValue:
    """
    Wraps the trained network for policy + value inference.
    """
    def __init__(self, model_path: str, device: torch.device, move_vocab: MoveVocab):
        self.device = device
        self.model = HiveNet(
            in_channels=IN_CHANNELS,
            board_size=(GRID_SIZE, GRID_SIZE),
            move_space_size=MOVE_SPACE_SIZE,
            reserve_vec_len=RESERVE_VEC_LEN
        ).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.vocab = move_vocab

    @torch.no_grad()
    def infer(self, hive) -> Tuple[Dict[Tuple, float], float]:
        _, color = active_player_and_color(hive)
        turn_number = getattr(hive, "turn", 0)
        board_state, reserves = get_board_state_and_reserves(hive)

        state_tensor, reserve_vec = encode_board(board_state, color, turn_number, reserves=reserves)
        state_tensor = state_tensor.unsqueeze(0).to(self.device)
        reserve_vec = reserve_vec.unsqueeze(0).to(self.device)

        policy_logits, value = self.model(state_tensor, reserve_vec)
        policy_logits = policy_logits[0]
        value = value[0].item()

        legal = list_legal_moves(hive, color)

        # Map legal moves to policy indices via vocab; mask illegal/unmapped
        scores = []
        idxs = []
        mapped_moves = []
        for mv in legal:
            mv_str = move_tuple_to_string(mv, hive)
            idx = self.vocab.to_index(mv_str)
            if idx is None or idx >= policy_logits.numel():
                continue
            idxs.append(idx)
            scores.append(policy_logits[idx].item())
            mapped_moves.append(mv)

        priors: Dict[Tuple, float] = {}
        if len(scores) == 0:
            # Fallback to uniform priors over legal moves
            if len(legal) == 0:
                return {}, value
            prob = 1.0 / len(legal)
            priors = {mv: prob for mv in legal}
            return priors, value

        # Softmax over mapped legal move logits
        logits = torch.tensor(scores, dtype=torch.float32, device=self.device)
        probs = torch.softmax(logits, dim=0).cpu().numpy().tolist()
        for mv, p in zip(mapped_moves, probs):
            priors[mv] = float(p)
        # If some legal moves didn’t map, give them a tiny floor so MCTS can still explore
        unmapped = [mv for mv in legal if mv not in priors]
        if unmapped:
            eps = 1e-6
            remain = max(0.0, 1.0 - sum(priors.values()) - eps * len(unmapped))
            # scale down existing priors to leave room for eps
            if sum(priors.values()) > 0:
                scale = max(1e-6, (sum(priors.values())))
                for k in list(priors.keys()):
                    priors[k] *= max(1e-6, (remain / scale))
            for mv in unmapped:
                priors[mv] = eps
            # renormalize
            s = sum(priors.values())
            if s > 0:
                for k in list(priors.keys()):
                    priors[k] /= s

        return priors, value

# -----------------------
# MCTS with PUCT
# -----------------------

class MCTSNode:
    def __init__(self, hive, priors: Dict[Tuple, float], to_play: str):
        self.hive = hive  # game state at this node
        self.to_play = to_play
        self.children: Dict[Tuple, "MCTSNode"] = {}
        self.priors = priors  # move -> P
        self.N: Dict[Tuple, int] = {m: 0 for m in priors}   # visits per move
        self.W: Dict[Tuple, float] = {m: 0.0 for m in priors}  # total value
        self.Q: Dict[Tuple, float] = {m: 0.0 for m in priors}  # mean value

    def total_visits(self) -> int:
        return sum(self.N.values())

    def select(self, c_puct: float) -> Tuple[Tuple, "MCTSNode"]:
        """
        Select a move by maximizing Q + U.
        """
        sum_n = self.total_visits()
        best_move = None
        best_score = -1e9
        for m in self.priors:
            q = self.Q[m]
            p = self.priors[m]
            n = self.N[m]
            u = c_puct * p * math.sqrt(sum_n + 1e-8) / (1 + n)
            s = q + u
            if s > best_score:
                best_score = s
                best_move = m
        return best_move, self.children.get(best_move)

    def expand(self, move: Tuple, child_hive, child_priors: Dict[Tuple, float], child_to_play: str) -> "MCTSNode":
        node = MCTSNode(child_hive, child_priors, child_to_play)
        self.children[move] = node
        if move not in self.N:
            self.N[move] = 0
            self.W[move] = 0.0
            self.Q[move] = 0.0
            # if we didn’t have a prior for this move, give small prior
            self.priors[move] = self.priors.get(move, 1e-6)
        return node

    def backup(self, move_path: List[Tuple], value_from_root_player: float):
        """
        Back up the value along the selected path.
        value_from_root_player is already aligned to the root player's perspective.
        """
        for parent, move in move_path:
            parent.N[move] += 1
            parent.W[move] += value_from_root_player
            parent.Q[move] = parent.W[move] / parent.N[move]

class TDMCTSAgent:
    def __init__(self,
                 model_path: str = "hive_model.pth",
                 num_simulations: int = 200,
                 c_puct: float = 2.0,
                 add_dirichlet_noise: bool = True,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_eps: float = 0.25,
                 device: Optional[torch.device] = None,
                 move_vocab_path: Optional[str] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab = MoveVocab(move_vocab_path)
        self.net = TDPolicyValue(model_path, self.device, self.vocab)
        self.num_sim = num_simulations
        self.c_puct = c_puct
        self.add_noise = add_dirichlet_noise
        self.alpha = dirichlet_alpha
        self.eps = dirichlet_eps

    def _root_node(self, hive) -> MCTSNode:
        priors, _ = self.net.infer(hive)
        # Optional Dirichlet noise at root
        if self.add_noise and priors:
            moves = list(priors.keys())
            noise = list(torch.distributions.Dirichlet(torch.full((len(moves),), self.alpha)).sample().cpu().numpy())
            for mv, eta in zip(moves, noise):
                priors[mv] = (1 - self.eps) * priors[mv] + self.eps * float(eta)
            # renormalize
            s = sum(priors.values())
            if s > 0:
                for mv in moves:
                    priors[mv] /= s
        _, color = active_player_and_color(hive)
        return MCTSNode(clone_hive(hive), priors, color)

    def _simulate(self, root: MCTSNode):
        node = root
        path: List[Tuple[MCTSNode, Tuple]] = []

        # Selection
        while True:
            terminal, value = is_terminal(node.hive)
            if terminal:
                # Value is from player-to-move at node; convert to root player's perspective
                val = value if node.to_play == root.to_play else -value
                node.backup(path, val)
                return

            if not node.priors or len(node.priors) == 0:
                # No prior (e.g., no legal moves?) -> treat as terminal draw
                node.backup(path, 0.0)
                return

            move, child = node.select(self.c_puct)
            path.append((node, move))

            # If child exists, descend; else expand
            if child is not None:
                node = child
                continue

            # Expand
            child_hive = clone_hive(node.hive)
            ok = apply_move_in_place(child_hive, move)
            if not ok:
                # Illegal application; give it zero prior/ban by backing with bad value
                node.N[move] += 1
                node.Q[move] = node.W[move] / node.N[move]
                # back up a slight penalty
                node.backup(path, -0.01)
                return

            priors, value = self.net.infer(child_hive)
            # Next player to move
            _, child_color = active_player_and_color(child_hive)
            child_node = node.expand(move, child_hive, priors, child_color)

            # Backup leaf value to root perspective
            val = value if child_node.to_play != root.to_play else -value
            child_node.backup(path, val)
            return

    def select_move(self, hive):
        """
        Return a move tuple in the same format as RandomAgent.select_move:
        ('play'/'move', (actPiece, refPiece, direction)) or ('non_play','pass')
        """
        root = self._root_node(hive)

        for _ in range(self.num_sim):
            self._simulate(root)

        # Select move with highest visit count
        if not root.N:
            # Fallback: try any legal move
            _, color = active_player_and_color(hive)
            legal = list_legal_moves(hive, color)
            return legal[0] if legal else ('non_play', 'pass')

        best_move = max(root.N.items(), key=lambda kv: kv[1])[0]
        return best_move

# --- Missing helpers added here ---

def clone_hive(src: Hive) -> Hive:
    """
    Create a deep copy of a Hive position (sufficient for MCTS).
    """
    h = Hive()
    h.turn = src.turn
    h.activePlayer = src.activePlayer
    h.players = src.players[:]

    # Board
    h.board.board = copy.deepcopy(src.board.board)
    h.board.ref0x = src.board.ref0x
    h.board.ref0y = src.board.ref0y

    # Played pieces
    h.playedPieces = {}
    for name, info in src.playedPieces.items():
        p_src = info['piece']
        p_clone = HivePiece(p_src.color, p_src.kind, p_src.number)
        h.playedPieces[name] = {'piece': p_clone, 'cell': info['cell']}

    # Pieces in cells (list of names)
    h.piecesInCell = {cell: stack[:] for cell, stack in src.piecesInCell.items()}

    # Unplayed pieces
    h.unplayedPieces = {}
    for color, dct in src.unplayedPieces.items():
        h.unplayedPieces[color] = {}
        for name, piece in dct.items():
            h.unplayedPieces[color][name] = HivePiece(piece.color, piece.kind, piece.number)

    return h


def apply_move_in_place(hive: Hive, move: Tuple) -> bool:
    """
    Apply a move tuple to hive in-place.
    move formats:
        ('play', (actPiece, refPiece, direction))
        ('move', (actPiece, refPiece, direction))  # engine distinguishes internally
        ('non_play', 'pass')
    Returns True if applied, False if illegal.
    """
    try:
        if not move:
            return False
        if move[0] == 'non_play':
            hive.action('non_play', 'pass')
            return True
        kind, (actPiece, refPiece, direction) = move
        # Hive.action always called with 'play'; it decides place vs move based on piece state
        hive.action('play', (actPiece, refPiece, direction))
        return True
    except Exception:
        return False