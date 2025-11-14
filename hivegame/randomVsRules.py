import os
from tqdm import tqdm
from hivegame.hive import Hive, HiveException
from hivegame.piece import HivePiece
from hivegame.agent import RandomAgent
from hivegame.rules_agent import RulesAgent
from hivegame.view import HiveView

MAX_MOVES_PER_GAME = 300

def piece_set(color):
    pieceSet = {}
    for i in range(3):
        ant = HivePiece(color, 'A', i+1)
        pieceSet[str(ant)] = ant
        grasshopper = HivePiece(color, 'G', i+1)
        pieceSet[str(grasshopper)] = grasshopper
    for i in range(2):
        spider = HivePiece(color, 'S', i+1)
        pieceSet[str(spider)] = spider
        beetle = HivePiece(color, 'B', i+1)
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
            ref_cell = hive.locate(refPiece)
            direction = {
                '|*': hive.W,
                '/*': hive.NW,
                '*\\': hive.NE,
                '*|': hive.E,
                '*/': hive.SE,
                '\\*': hive.SW,
                '=*': hive.O
            }[pointOfContact]
            target_cell = hive._poc2cell(refPiece, direction)
            if start_cell == target_cell:
                return False
        if pointOfContact is None and turn > 1:
            return False
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

def play_one_game(game_idx, save_dir, rules_as_white):
    hive = Hive()
    player = {1: piece_set('w'), 2: piece_set('b')}
    if rules_as_white:
        agent = {1: RulesAgent('w'), 2: RandomAgent('b')}
        agent_names = {1: "RulesAgent", 2: "RandomAgent"}
    else:
        agent = {1: RandomAgent('w'), 2: RulesAgent('b')}
        agent_names = {1: "RandomAgent", 2: "RulesAgent"}
    hive.turn += 1  # white player start
    hive.setup()
    move_log = []
    moves=0 # move counter to prevent infinite games

    while hive.check_victory() == hive.UNFINISHED and moves < MAX_MOVES_PER_GAME:
        active_player = (2 - (hive.turn % 2))
        move = agent[active_player].select_move(hive)
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

        move_log.append(cmd)
        exec_cmd(hive, player[active_player], cmd, hive.turn)
        moves += 1


    result = hive.check_victory()
    if moves >= MAX_MOVES_PER_GAME and result == hive.UNFINISHED:
        result = hive.DRAW

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"game_{game_idx}.txt"), "w") as f:
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

def main():
    save_dir = "randomVrRulesGamesRecord"
    num_games_each = 100
    rules_wins = 0
    random_wins = 0
    draws = 0

    total_games = num_games_each * 2
    pbar = tqdm(total=total_games, desc="Playing games")

    # RulesAgent as White
    for i in range(num_games_each):
        result, agent_names = play_one_game(i+1, save_dir, rules_as_white=True)
        if result == Hive.WHITE_WIN:
            rules_wins += 1
        elif result == Hive.BLACK_WIN:
            random_wins += 1
        elif result == Hive.DRAW:
            draws += 1
        pbar.update(1)

    # RulesAgent as Black
    for i in range(num_games_each):
        result, agent_names = play_one_game(i+1+num_games_each, save_dir, rules_as_white=False)
        if result == Hive.BLACK_WIN:
            rules_wins += 1
        elif result == Hive.WHITE_WIN:
            random_wins += 1
        elif result == Hive.DRAW:
            draws += 1
        pbar.update(1)

    pbar.close()
    print(f"RulesAgent win rate: {rules_wins/total_games:.2f}")
    print(f"RandomAgent win rate: {random_wins/total_games:.2f}")
    print(f"Draw rate: {draws/total_games:.2f}")

if __name__ == "__main__":
    main()


    