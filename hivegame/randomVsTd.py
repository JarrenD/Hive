#!/usr/bin/env python3
"""
randomVsTd.py

Pit your RandomAgent against the TD value-network agent (HiveValueNet).
This mirrors the structure of your randomVsRules.py script so results and logs
are saved in the same format.

Usage:
    python randomVsTd.py --model hive_value_model.pth --num-games 25 --save-dir randomVstdGamesRecord
"""
import os
import random
from tqdm import tqdm
import argparse
import warnings

from hivegame.hive import Hive
from hivegame.piece import HivePiece
from hivegame.agent import RandomAgent

# Import the TD value net and helper parsing functions
from hivegame.hivenet_td import make_value_model, HiveValueNet
from hivegame.hivenet_train import _parse_cmd, _poc_to_direction

MAX_MOVES_PER_GAME = 500  # or 500, or whatever is reasonable

# -----------------------
# Helpers (copied from your original harness)
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


def exec_cmd(hive, player, cmd, turn):
    """
    Execute command string on given hive using the provided player mapping.
    Returns True if the command applied cleanly, False otherwise (no state change on failure).
    This function mirrors the semantics in your other harnesses.
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
            # attempt to use _poc2cell if available
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

        # lookup piece object from player's piece_set; if key missing this will throw and be caught
        p = player[actPiece]
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
# TD value-agent wrapper (returns moves in the same shape as other agents)
# -----------------------
class ValueAgent:
    """
    Wraps a HiveValueNet model so that .select_move(hive) returns a move tuple
    compatible with the harness (('play',(actPiece, refPiece, direction)) or ('non_play','pass')).
    """

    def __init__(self, color: str, model: HiveValueNet, device=None):
        self.color = color
        self.model = model
        self.device = device if device is not None else ("cuda" if hasattr(model, "to") and 
                                                         False else "cpu")  # actual device set later

    def select_move(self, hive):
        """
        Query the model for a command string (CLI style) and convert it to the expected tuple format.
        """
        # ensure model and device are ready
        device = next(self.model.parameters()).device if any(True for _ in self.model.parameters()) else None
        # select_move_by_value is an instance method implemented on HiveValueNet
        try:
            cmd = self.model.select_move_by_value_subset(hive, self.color, hive.turn, candidate_k=25, device=device, explore_eps=0.0)
        except Exception as e:
            # If model call fails for some reason, fallback to pass
            warnings.warn(f"ValueAgent.select_move: model.select_move_by_value_subset raised: {e}; falling back to 'pass'")
            return ('non_play', 'pass')

        # parse the CLI-style command into the standard tuple form
        parsed = _parse_cmd(cmd)  # from hivenet_train: returns ('non_play','pass') or ('play',(movingPiece, pointOfContact, refPiece))
        if parsed[0] == 'non_play':
            return ('non_play', 'pass')

        # parsed[0] == 'play'
        movingPiece, pointOfContact, refPiece = parsed[1]
        if pointOfContact is None:
            # first-placement style: ('play', (actPiece, None, None))
            return ('play', (movingPiece, None, None))

        # convert POC string (e.g. '|*') to engine direction constant using hive instance
        try:
            direction = _poc_to_direction(hive, pointOfContact)
        except Exception:
            # if conversion fails, return pass to be safe
            warnings.warn(f"ValueAgent.select_move: failed to convert POC '{pointOfContact}' to direction; falling back to 'pass'")
            return ('non_play', 'pass')

        return ('play', (movingPiece, refPiece, direction))


# -----------------------
# Game runner (mirrors randomVsRules but uses ValueAgent instead of RulesAgent)
# -----------------------
def play_one_game(game_idx, save_dir, td_as_white, model: HiveValueNet, device):
    hive = Hive()
    player = {1: piece_set('w'), 2: piece_set('b')}

    if td_as_white:
        agent = {1: ValueAgent('w', model, device), 2: RandomAgent('b')}
        agent_names = {1: "ValueAgent", 2: "RandomAgent"}
    else:
        agent = {1: RandomAgent('w'), 2: ValueAgent('b', model, device)}
        agent_names = {1: "RandomAgent", 2: "ValueAgent"}

    hive.turn += 1  # white player start
    hive.setup()
    move_log = []
    moves = 0  # Add move counter

    # Play until game ends or move limit reached
    while hive.check_victory() == hive.UNFINISHED and moves < MAX_MOVES_PER_GAME:
        active_player = (2 - (hive.turn % 2))
        move = agent[active_player].select_move(hive)

        # Convert agent move tuple into CLI string (same logic as your original harness)
        #print(move)
        if move[0] == 'play':
            actPiece, refPiece, direction = move[1]
            cmd = actPiece
            if refPiece is not None and direction is not None:
                poc = direction_to_poc(direction, hive)
                cmd = f"{actPiece}{poc}{refPiece}"
        elif move[0] == 'move':
            actPiece, refPiece, direction = move[1]
            poc = direction_to_poc(direction, hive)
            cmd = f"{actPiece}{poc}{refPiece}"
        elif move[0] == 'non_play':
            cmd = 'pass'
        else:
            cmd = 'pass'  # fallback
            

        # Append to log then execute (we try to ensure illegal moves are handled)
        move_log.append(cmd)
        ok = exec_cmd(hive, player[active_player], cmd, hive.turn)
        if not ok:
            # model produced an illegal move somehow. Try to recover by playing a random legal move.
            try:
                valid_moves = hive.find_valid_moves('w' if active_player == 1 else 'b')
            except Exception:
                valid_moves = []
            if valid_moves:
                fallback_cmd = random.choice(valid_moves)
                warnings.warn(f"Model produced illegal move '{cmd}' — falling back to random legal move '{fallback_cmd}'")
                exec_cmd(hive, player[active_player], fallback_cmd, hive.turn)
                move_log[-1] = fallback_cmd  # replace last logged move for correctness
            else:
                # if no legal moves found, force pass
                warnings.warn(f"Model produced illegal move '{cmd}' and no legal fallback moves found. Passing.")
                exec_cmd(hive, player[active_player], 'pass', hive.turn)
                move_log[-1] = 'pass'
        moves += 1  # Increment move counter

    # If move limit reached, treat as draw
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
# CLI & main loop
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="hive_value_model.pth", help="Path to value model checkpoint")
    parser.add_argument("--num-games", type=int, default=100, help="Number of games per side (total games = 2 * num-games)")
    parser.add_argument("--save-dir", type=str, default="randomVstdGamesRecord", help="Directory to write game logs")
    parser.add_argument("--device", type=str, default=None, help="Device for model (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    device = args.device if args.device else ("cuda" if __import__("torch").cuda.is_available() else "cpu")

    # load model
    model = make_value_model()
    try:
        model.load_state_dict(__import__("torch").load(args.model, map_location=device))
    except Exception as e:
        raise RuntimeError(f"Failed to load model checkpoint '{args.model}': {e}")
    model.to(device)
    model.eval()

    num_games_each = args.num_games
    rules_wins = 0
    random_wins = 0
    draws = 0

    total_games = num_games_each * 2
    pbar = tqdm(total=total_games, desc="Playing games")

    # ValueAgent as White
    for i in range(num_games_each):
        result, agent_names = play_one_game(i + 1, args.save_dir, td_as_white=True, model=model, device=device)
        if result == Hive.WHITE_WIN:
            # ValueAgent is white in this block
            random_wins += 0  # placeholder for clarity
            # which side won? agent_names[1] indicates
            if agent_names[1] == "ValueAgent":
                # Value agent won
                pass  # we'll count below uniformly
        if result == Hive.WHITE_WIN:
            # count wins properly
            if agent_names[1] == "ValueAgent":
                random_wins += 0  # not relevant here
        # update tallies consistent with original script:
        if result == Hive.WHITE_WIN:
            if agent_names[1] == "ValueAgent":
                td_win = True
                # ValueAgent as white won -> increment td wins
                td_winner = "td"
            else:
                td_winner = "random"
            # simpler: use agent_names mapping to determine
        # we will compute tally after both halves in the simpler block below
        pbar.update(1)

    # ValueAgent as Black
    for i in range(num_games_each):
        result, agent_names = play_one_game(i + 1 + num_games_each, args.save_dir, td_as_white=False, model=model, device=device)
        pbar.update(1)

    pbar.close()

    # Now scan saved games to compute exact tallies (safer & simpler)
    rules_wins = 0
    random_wins = 0
    draws = 0

    # read logs written to save_dir
    files = sorted([f for f in os.listdir(args.save_dir) if f.startswith("game_") and f.endswith(".txt")])
    for fname in files:
        with open(os.path.join(args.save_dir, fname), "r") as f:
            txt = f.read()
        if "White wins!" in txt:
            # find which agent was white in that file by parsing parentheses if present
            if "(ValueAgent)" in txt:
                # Value agent was white and won
                td_won = True
                if "(RandomAgent)" in txt:
                    # ambiguous but unlikely
                    pass
                # increment td or random based on which agent string present
                if "(ValueAgent)" in txt:
                    # check the winning line to see which agent was reported as winner
                    # The original format writes: "White wins! (AgentName)"
                    # so find the line with "White wins!"
                    for line in txt.splitlines():
                        if line.startswith("Result:"):
                            continue
                        if "White wins!" in line:
                            # extract agent name in parentheses
                            idx = line.find("(")
                            if idx != -1:
                                agent = line[idx + 1:line.find(")", idx)]
                                if agent == "ValueAgent":
                                    td_wins = 1
                                elif agent == "RandomAgent":
                                    random_wins += 1
                            else:
                                # fallback increment
                                random_wins += 0
            else:
                # fallback: assume RandomAgent
                random_wins += 1
        elif "Black wins!" in txt:
            if "(ValueAgent)" in txt:
                td_wins = td_wins + 1 if 'td_wins' in locals() else 1
            else:
                random_wins += 1
        elif "Draw" in txt:
            draws += 1

    # If parsing above failed to compute totals correctly (it is conservative), compute quick estimate:
    # Count how many files contain "(ValueAgent)" and whether the winner field contains that string.
    # Simpler: scan each file's Result line
    td_wins = 0
    random_wins = 0
    draws = 0
    for fname in files:
        with open(os.path.join(args.save_dir, fname), "r") as f:
            lines = f.readlines()
        # find Result line
        for ln in lines[::-1]:
            if ln.startswith("Result:"):
                res_line = ln
                break
        else:
            res_line = ""
        # locate agent parentheses on the same line or previous lines:
        # our writer writes winner and parentheses on same line, so parse it
        if "White wins!" in res_line:
            if "(ValueAgent)" in res_line:
                td_wins += 1
            elif "(RandomAgent)" in res_line:
                random_wins += 1
            else:
                # unknown, try to heuristically find agent in file
                if "(ValueAgent)" in "".join(lines):
                    td_wins += 1
                else:
                    random_wins += 1
        elif "Black wins!" in res_line:
            if "(ValueAgent)" in res_line:
                td_wins += 1
            elif "(RandomAgent)" in res_line:
                random_wins += 1
            else:
                if "(ValueAgent)" in "".join(lines):
                    td_wins += 1
                else:
                    random_wins += 1
        elif "Draw" in res_line:
            draws += 1

    total_games = len(files)
    print(f"TD (ValueAgent) win rate: {td_wins/total_games:.2f} ({td_wins}/{total_games})")
    print(f"RandomAgent win rate: {random_wins/total_games:.2f} ({random_wins}/{total_games})")
    print(f"Draw rate: {draws/total_games:.2f} ({draws}/{total_games})")


if __name__ == "__main__":
    main()
