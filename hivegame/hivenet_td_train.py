"""
hivenet_td_train.py (iterative, memory-safe)

Iterative self-play + TD training harness for the Hive value network.

Major changes for WSL stability:
  • Online TD targets at collection time (store only (state, reserve, target) — no next-state in replay)
  • Replay stored in float16 to halve RAM
  • Always *sampled* move evaluation (no full enumeration); small candidate set by default
  • Small default replay size; frequent GC; optional CUDA empty cache

Usage (example):
    python -m hivegame.hivenet_td_train \
        --cycles 20 --games-per-cycle 8 --train-steps-per-cycle 200 --model-out hive_value_model.pth

The defaults are conservative; scale up gradually.
"""
from __future__ import annotations
import argparse
import copy
import random
import gc
from collections import deque
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Project imports
from hivegame.hive import Hive
from hivegame.hivenet_td import HiveValueNet, make_value_model, _apply_move_on_clone_flexible
from hivegame.hivenet_train import (
    _current_board_state_from_hive,
    _reserves_from_hive,
    encode_board,
    _apply_move_on_hive,
)


# ------------------ Replay buffer (compact) ------------------
class ReplayBuffer:
    """Stores (state_tensor_half, reserve_vec_half, td_target_float32)."""

    def __init__(self, capacity: int = 8192):
        self.buf = deque(maxlen=capacity)

    def push(self, s_tensor: torch.Tensor, s_res: torch.Tensor, td_target: float) -> None:
        # store as float16 to reduce memory usage significantly
        self.buf.append((s_tensor.detach().cpu().half(), s_res.detach().cpu().half(), float(td_target)))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s_half, sr_half, tgt = zip(*batch)
        s = torch.stack(s_half)  # float16 on CPU
        sr = torch.stack(sr_half)  # float16 on CPU
        tgt = torch.tensor(tgt, dtype=torch.float32)  # (B,)
        return s, sr, tgt

    def __len__(self) -> int:
        return len(self.buf)


# ------------------ Candidate evaluation helper ------------------

def evaluate_candidate_subset_and_choose(
    model: HiveValueNet,
    hive: Hive,
    color: str,
    turn: int,
    legal_moves: List,
    candidate_k: int,
    device: torch.device,
    eval_batch_size: int = 64,
) -> str:
    """Sample up to candidate_k legal moves, evaluate successors, return the best."""
    if not legal_moves:
        return "pass"

    # sample small candidate set to limit deepcopy/encoding cost
    k = min(candidate_k, len(legal_moves))
    candidates = random.sample(legal_moves, k) if len(legal_moves) > k else list(legal_moves)

    succ_states, succ_res, succ_moves = [], [], []
    for mv in candidates:
        try:
            h_clone = copy.deepcopy(hive)
            _apply_move_on_clone_flexible(h_clone, mv, turn)
        except Exception:
            continue

        # encode successor (after move): side flips, turn increments
        next_color = "w" if color == "b" else "b"
        bs = _current_board_state_from_hive(h_clone)
        rs = _reserves_from_hive(h_clone)
        st_tensor, res_vec = encode_board(bs, next_color, turn + 1, reserves=rs)
        succ_states.append(st_tensor)
        succ_res.append(res_vec)
        succ_moves.append(mv)
        del h_clone

    if not succ_moves:
        return random.choice(list(legal_moves))

    X_states = torch.stack(succ_states, dim=0)
    X_res = torch.stack(succ_res, dim=0)
    vals = model.evaluate_successors(X_states, X_res, device=device, batch_size=eval_batch_size)

    max_v = max(vals)
    best_idxs = [i for i, v in enumerate(vals) if v == max_v]
    choice_idx = random.choice(best_idxs) if len(best_idxs) > 1 else best_idxs[0]
    return succ_moves[choice_idx]


# ------------------ Self-play with *online* TD targets ------------------

def self_play_episode(
    model: HiveValueNet,
    target: HiveValueNet,
    device: torch.device,
    gamma: float,
    max_moves: int = 500,
    epsilon: float = 0.35,
    candidate_k: int = 12,
) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
    """
    Play a self-play game using epsilon-greedy over value estimates.
    Returns a list of (state_tensor, reserve_vec, td_target) with bootstrap computed using the *target* net.
    This avoids storing next-state tensors in the replay buffer.
    """
    model.to(device).eval()
    target.to(device).eval()

    game = Hive()
    game.turn += 1
    game.setup()

    samples: List[Tuple[torch.Tensor, torch.Tensor, float]] = []
    moves = 0

    while True:
        if game.check_victory() != game.UNFINISHED:
            break
        if moves >= max_moves:
            done = True
            reward = -0.1
            samples.append((s_tensor, s_res, float(reward)))
            break

        active = 1 if game.turn % 2 == 1 else 2
        color = "w" if active == 1 else "b"

        # encode current state (before move)
        bs = _current_board_state_from_hive(game)
        rs = _reserves_from_hive(game)
        s_tensor, s_res = encode_board(bs, color, game.turn, reserves=rs)

        # legal moves
        try:
            legal = list(game.find_valid_moves(color))
        except Exception:
            legal = []

        if not legal:
            mv = "pass"
        else:
            # epsilon-greedy exploration
            if random.random() < epsilon:
                mv = random.choice(legal)
            else:
                mv = evaluate_candidate_subset_and_choose(
                    model, game, color, game.turn, legal, candidate_k, device
                )

        # apply on the real game
        try:
            _apply_move_on_hive(game, mv, game.turn)
        except Exception:
            # if illegal somehow, try pass
            try:
                _apply_move_on_hive(game, "pass", game.turn)
            except Exception:
                pass

        # next state
        next_active = 1 if game.turn % 2 == 1 else 2
        next_color = "w" if next_active == 1 else "b"
        bs_n = _current_board_state_from_hive(game)
        rs_n = _reserves_from_hive(game)
        s_next, s_next_res = encode_board(bs_n, next_color, game.turn, reserves=rs_n)

        # reward from mover's perspective
        done = (game.check_victory() != game.UNFINISHED)
        reward = 0.0
        if done:
            res = game.check_victory()
            if res == game.WHITE_WIN:
                reward = 1.0 if color == "w" else -1.0
            elif res == game.BLACK_WIN:
                reward = 1.0 if color == "b" else -1.0
            else:
                reward = 0.0

        with torch.no_grad():
            v_next = target(s_next.unsqueeze(0).to(device), s_next_res.unsqueeze(0).to(device)).item()
        td_target = reward if done else (reward + gamma * v_next)

        samples.append((s_tensor, s_res, float(td_target)))
        moves += 1

    # GC between games
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    return samples


# ------------------ Single training step ------------------

def train_one_step(
    model: HiveValueNet,
    optimizer: optim.Optimizer,
    replay: ReplayBuffer,
    batch_size: int,
    device: torch.device,
) -> Optional[float]:
    if len(replay) < batch_size:
        return None

    s_half, sr_half, tgt = replay.sample(batch_size)
    # move to device and cast to float32 for the model
    s = s_half.to(torch.float32).to(device)
    sr = sr_half.to(torch.float32).to(device)
    tgt = tgt.to(device)

    model.train()
    pred = model(s, sr)
    loss = nn.MSELoss()(pred, tgt)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.item())


# ------------------ main iterative loop ------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=20, help="Number of selfplay->train cycles")
    parser.add_argument("--games-per-cycle", type=int, default=200, help="Self-play games per cycle")
    parser.add_argument("--train-steps-per-cycle", type=int, default=2000, help="Gradient updates per cycle")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor for TD target")
    parser.add_argument("--replay-capacity", type=int, default=131072, help="Replay buffer capacity (small to protect RAM)")
    parser.add_argument("--epsilon", type=float, default=0.35, help="Epsilon for epsilon-greedy self-play")
    parser.add_argument("--candidate-k", type=int, default=30, help="#candidate legal moves evaluated per decision")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu (auto if omitted)")
    parser.add_argument("--model-out", type=str, default="hive_value_model.pth", help="Checkpoint path (overwritten each cycle)")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[info] using device: {device}")

    model = make_value_model()
    target = copy.deepcopy(model)
    model.to(device)
    target.to(device)
    target.eval()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    replay = ReplayBuffer(capacity=args.replay_capacity)
    total_grad_steps = 0

    for cycle in range(1, args.cycles + 1):
        print(f"=== Cycle {cycle}/{args.cycles} ===")

        # --- 1) Self-play collection with online TD target ---
        print(f"[cycle {cycle}] generating {args.games_per_cycle} self-play games (epsilon={args.epsilon}, candidates={args.candidate_k})")
        pbar = tqdm(range(args.games_per_cycle), desc="Self-play", leave=False)
        for _ in pbar:
            samples = self_play_episode(
                model,
                target,
                device,
                gamma=args.gamma,
                max_moves=500,
                epsilon=args.epsilon,
                candidate_k=args.candidate_k,
            )
            for (s, sr, tgt) in samples:
                replay.push(s, sr, tgt)
            pbar.set_postfix(replay_len=len(replay))

        print(f"[cycle {cycle}] replay size: {len(replay)} (capacity {args.replay_capacity})")

        # --- 2) Training over replay ---
        print(f"[cycle {cycle}] training for {args.train_steps_per_cycle} steps (batch_size={args.batch_size})")
        train_pbar = tqdm(range(args.train_steps_per_cycle), desc="Train", leave=False)
        for _ in train_pbar:
            loss = train_one_step(model, optimizer, replay, args.batch_size, device)
            if loss is not None:
                total_grad_steps += 1
                # softly update target every N steps (Polyak-style could also be used)
                if total_grad_steps % 400 == 0:
                    target.load_state_dict(model.state_dict())
                train_pbar.set_postfix(loss=loss)

        # hard update and checkpoint
        target.load_state_dict(model.state_dict())
        torch.save(model.state_dict(), args.model_out)
        print(f"[cycle {cycle}] checkpoint saved to {args.model_out}")

        # housekeeping between cycles
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    print("[info] training complete.")


if __name__ == "__main__":
    main()
