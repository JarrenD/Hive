#!/usr/bin/env python3
"""
randomVsMcts.py

Pit your RandomAgent against the MCTS+TD policy/value agent (TDMCTSAgent).
Saves logs in the same text format as your other harnesses.

Usage:
    python randomVsMcts.py --model hive_model.pth --num-games 25 --save-dir randomVsMctsGamesRecord \
                           --sims 300 --c-puct 2.0 --move-vocab move_vocab.json
"""
import os
import random
from tqdm import tqdm
import argparse
import warnings

from hivegame.hive import Hive
from hivegame.piece import HivePiece
from hivegame.agent import RandomAgent

from hivegame.hivenet_mcts import TDMCTSAgent

MAX_MOVES_PER_GAME = 500

# -----------------------
# Helpers
# -----------------------
def piece_set(color):
    pieceSet = {}
    for i in range(3):
        ant = HivePiece(color, 'A', i + 1)
        pieceSet[str(ant)] = ant
        grasshopper = HivePiece(color, 'G', i + 1)
        pieceSet[str(grasshopper)] = grasshopper
    for i in range(2):
        spider = HivePiece(color, 'S', i + 1)
        pieceSet[str(spider)] = spider
        beetle = HivePiece(color, 'B', i + 1)
        pieceSet[str(beetle)] = beetle
    queen = HivePiece(color, 'Q', 1)
    pieceSet[str(queen)] = queen
    return pieceSet

def direction_to_poc(direction, hive):
    dir_map = {
        hive.W: '|*',
        hive.NW: '/*',
        hive.NE: '*\\',
        hive.E: '*|',
        hive.SE: '*/',
        hive.SW: '\\*',
        hive.O: '=*'
    }
    return dir_map.get(direction, '')

def move_tuple_to_cmd(hive, move_tuple):
    """
    Convert ('play'/'move', (actPiece, refPiece, direction)) or ('non_play','pass') to CLI string.
    """
    if not move_tuple:
        return 'pass'
    if move_tuple[0] == 'non_play':
        return 'pass'
    kind, (actPiece, refPiece, direction) = move_tuple
    if refPiece is None or direction is None:
        return actPiece
    poc = direction_to_poc(direction, hive)
    return f"{actPiece}{poc}{refPiece}"

def exec_cmd(hive, player, cmd, turn):
    """
    Apply a CLI-style command string to the engine (same semantics as your other harness).
    """
    try:
        if cmd == 'pass':
            hive.action('non_play', cmd)
            return True

        if len(cmd) == 3:
            actPiece = cmd
            pointOfContact = None
            refPiece = None
        else:
            if len(cmd) != 8:
                return False
            actPiece = cmd[:3]
            pointOfContact = cmd[3:5]
            refPiece = cmd[5:]
            start_cell = hive.locate(actPiece)
            direction = {
                '|*': hive.W,
                '/*': hive.NW,
                '*\\': hive.NE,
                '*|': hive.E,
                '*/': hive.SE,
                '\\*': hive.SW,
                '=*': hive.O
            }[pointOfContact]
            target_cell = getattr(hive, "_poc2cell", lambda rp, d: None)(refPiece, direction)
            if start_cell is not None and target_cell is not None and start_cell == target_cell:
                return False

        if pointOfContact is None and turn > 1:
            return False

        # If playing from hand, ensure piece exists in player's pool
        _ = player[actPiece]
        direction = None
        if pointOfContact is not None:
            direction = {
                '|*': hive.W,
                '/*': hive.NW,
                '*\\': hive.NE,
                '*|': hive.E,
                '*/': hive.SE,
                '\\*': hive.SW,
                '=*': hive.O
            }[pointOfContact]
        hive.action('play', (actPiece, refPiece, direction))
    except Exception:
        return False
    return True

# -----------------------
# Game runner
# -----------------------
def play_one_game(game_idx, save_dir, mcts_as_white, mcts_agent: TDMCTSAgent):
    hive = Hive()
    player = {1: piece_set('w'), 2: piece_set('b')}

    if mcts_as_white:
        agent = {1: mcts_agent, 2: RandomAgent('b')}
        agent_names = {1: "MCTSAgent", 2: "RandomAgent"}
    else:
        agent = {1: RandomAgent('w'), 2: mcts_agent}
        agent_names = {1: "RandomAgent", 2: "MCTSAgent"}

    hive.turn += 1  # white player starts
    hive.setup()
    move_log = []
    moves = 0

    while hive.check_victory() == hive.UNFINISHED and moves < MAX_MOVES_PER_GAME:
        active_player = (2 - (hive.turn % 2))
        mv_tuple = agent[active_player].select_move(hive)  # already ('play'/... tuple)
        cmd = move_tuple_to_cmd(hive, mv_tuple)

        move_log.append(cmd)
        ok = exec_cmd(hive, player[active_player], cmd, hive.turn)
        if not ok:
            try:
                color = 'w' if active_player == 1 else 'b'
                valid_moves = hive.find_valid_moves(color) or []
            except Exception:
                valid_moves = []
            if valid_moves:
                fb_mv = random.choice(valid_moves)
                fb_cmd = move_tuple_to_cmd(hive, fb_mv)
                warnings.warn(f"Illegal move '{cmd}' — falling back to random legal '{fb_cmd}'")
                exec_cmd(hive, player[active_player], fb_cmd, hive.turn)
                move_log[-1] = fb_cmd
                cmd = fb_cmd  # update for logging
            else:
                warnings.warn(f"Illegal move '{cmd}' and no legal fallbacks. Passing.")
                exec_cmd(hive, player[active_player], 'pass', hive.turn)
                move_log[-1] = 'pass'
                cmd = 'pass'
        moves += 1

        # Progress print every 25 moves
        if moves % 25 == 0:
            print(f"[Game {game_idx}] Move {moves}: Player {agent_names[active_player]} "
                  f"({'w' if active_player == 1 else 'b'}) played '{cmd}' | turn={hive.turn} "
                  f"| playedPieces={len(hive.playedPieces)} | result={hive.check_victory()}")

    # If move cap reached, count as draw
    result = hive.check_victory()
    if moves >= MAX_MOVES_PER_GAME and result == hive.UNFINISHED:
        result = hive.DRAW

    # Save game log
    os.makedirs(save_dir, exist_ok=True)
    fname = os.path.join(save_dir, f"game_{game_idx}.txt")
    with open(fname, "w") as f:
        for m in move_log:
            f.write(m + "\n")
        f.write("\nResult: ")
        if result == hive.WHITE_WIN:
            winner = agent_names[1]
            f.write(f"White wins! ({winner})\n")
        elif result == hive.BLACK_WIN:
            winner = agent_names[2]
            f.write(f"Black wins! ({winner})\n")
        elif result == hive.DRAW:
            f.write("Draw!\n")
        else:
            f.write("Unknown result\n")

    return result, agent_names

# -----------------------
# CLI & main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="hive_model.pth", help="Path to policy/value model")
    parser.add_argument("--move-vocab", type=str, default=None, help="Path to move_vocab.json (optional)")
    parser.add_argument("--num-games", type=int, default=100, help="Games per side (total = 2 * num-games)")
    parser.add_argument("--save-dir", type=str, default="randomVsMctsGamesRecord", help="Directory for game logs")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    parser.add_argument("--sims", type=int, default=300, help="MCTS simulations per move")
    parser.add_argument("--c-puct", type=float, default=2.0, help="PUCT exploration constant")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3, help="Dirichlet alpha at root (if noise on)")
    parser.add_argument("--dirichlet-eps", type=float, default=0.25, help="Mix factor for root noise")
    parser.add_argument("--no-root-noise", action="store_true", help="Disable Dirichlet noise at root")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    device = args.device if args.device else ("cuda" if __import__("torch").cuda.is_available() else "cpu")

    # Build a single MCTS agent instance
    mcts_agent = TDMCTSAgent(
        model_path=args.model,
        num_simulations=args.sims,
        c_puct=args.c_puct,
        add_dirichlet_noise=not args.no_root_noise,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_eps=args.dirichlet_eps,
        device=device,
        move_vocab_path=args.move_vocab
    )

    td_wins = 0
    random_wins = 0
    draws = 0

    total_games = args.num_games * 2
    pbar = tqdm(total=total_games, desc="Playing games")

    # MCTS as White
    for i in range(args.num_games):
        result, agent_names = play_one_game(i + 1, args.save_dir, mcts_as_white=True, mcts_agent=mcts_agent)
        if result == Hive.WHITE_WIN and agent_names[1] == "MCTSAgent":
            td_wins += 1
        elif result == Hive.BLACK_WIN and agent_names[2] == "MCTSAgent":
            td_wins += 1
        elif result == Hive.WHITE_WIN and agent_names[1] == "RandomAgent":
            random_wins += 1
        elif result == Hive.BLACK_WIN and agent_names[2] == "RandomAgent":
            random_wins += 1
        else:
            if result == Hive.DRAW:
                draws += 1
        pbar.update(1)

    # MCTS as Black
    for i in range(args.num_games):
        result, agent_names = play_one_game(i + 1 + args.num_games, args.save_dir, mcts_as_white=False, mcts_agent=mcts_agent)
        if result == Hive.WHITE_WIN and agent_names[1] == "MCTSAgent":
            td_wins += 1
        elif result == Hive.BLACK_WIN and agent_names[2] == "MCTSAgent":
            td_wins += 1
        elif result == Hive.WHITE_WIN and agent_names[1] == "RandomAgent":
            random_wins += 1
        elif result == Hive.BLACK_WIN and agent_names[2] == "RandomAgent":
            random_wins += 1
        else:
            if result == Hive.DRAW:
                draws += 1
        pbar.update(1)

    pbar.close()

    total_played = td_wins + random_wins + draws
    total_played = max(total_played, 1)
    print(f"MCTSAgent win rate: {td_wins/total_played:.2f} ({td_wins}/{total_played})")
    print(f"RandomAgent win rate: {random_wins/total_played:.2f} ({random_wins}/{total_played})")
    print(f"Draw rate: {draws/total_played:.2f} ({draws}/{total_played})")

if __name__ == "__main__":
    main()