import sys
from hivegame.hive import Hive, HiveException
from hivegame.piece import HivePiece
from hivegame.agent import RandomAgent
from hivegame.view import HiveView

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

def parse_cmd(cmd):
    if cmd == 'pass':
        return ('non_play', cmd)
    if len(cmd) == 3:
        movingPiece = cmd
        pointOfContact = None
        refPiece = None
    else:
        if len(cmd) != 8:
            raise Exception("Failed to parse command.")
        movingPiece = cmd[:3]
        pointOfContact = cmd[3:5]
        refPiece = cmd[5:]
    return ('play', (movingPiece, pointOfContact, refPiece))

def poc2direction(pointOfContact):
    if pointOfContact == '|*':
        return Hive.W
    if pointOfContact == '/*':
        return Hive.NW
    if pointOfContact == '*\\':
        return Hive.NE
    if pointOfContact == '*|':
        return Hive.E
    if pointOfContact == '*/':
        return Hive.SE
    if pointOfContact == '\\*':
        return Hive.SW
    if pointOfContact == '=*':
        return Hive.O
    raise ValueError('Invalid input for point of contact: "%s"' % pointOfContact)

def exec_cmd(hive, player, cmd, turn):
    try:
        (cmdType, value) = parse_cmd(cmd)
        if cmdType == 'play':
            (actPiece, pointOfContact, refPiece) = value
        if cmdType == 'non_play' and value == 'pass':
            hive.action(cmdType, value)
            return True
    except:
        return False

    if pointOfContact is None and turn > 1:
        return False

    try:
        p = player[actPiece]
        direction = None
        if pointOfContact is not None:
            direction = poc2direction(pointOfContact)
    except:
        return False

    try:
        hive.action('play', (actPiece, refPiece, direction))
    except HiveException:
        return False
    return True

def main():
    hive = Hive()
    view = HiveView(hive)
    player = {1: piece_set('w'), 2: piece_set('b')}
    agent = RandomAgent('w')
    player2color = {1: 'w', 2: 'b'}
    hive.turn += 1  # white player start
    hive.setup()

    while hive.check_victory() == hive.UNFINISHED:
        print("Turn: %s" % hive.turn)
        active_player = (2 - (hive.turn % 2))
        print(view)
        print("pieces in hand: %s" % sorted(
            hive.unplayedPieces[player2color[active_player]]
        ))
        print("player %s play: " % active_player)
        if active_player == 1:
            # RandomAgent's turn
            move = agent.select_move(hive)
            if move[0] == 'play':
                actPiece, refPiece, direction = move[1]
                cmd = actPiece
                if refPiece is not None and direction is not None:
                    # Convert direction to pointOfContact string
                    dir_map = {
                        hive.W: '|*',
                        hive.NW: '/*',
                        hive.NE: '*\\',
                        hive.E: '*|',
                        hive.SE: '*/',
                        hive.SW: '\\*',
                        hive.O: '=*'
                    }
                    poc = dir_map.get(direction, '')
                    cmd = f"{actPiece}{poc}{refPiece}"
            elif move[0] == 'non_play':
                cmd = 'pass'
            print(f"AI plays: {cmd}")
        else:
            # Human's turn
            try:
                cmd = sys.stdin.readline().strip()
            except KeyboardInterrupt:
                break
        if exec_cmd(hive, player[active_player], cmd, hive.turn):
            print("\n")
            print("=" * 79)
            print("\n")
        else:
            print("invalid play!")

    print("\nThanks for playing Hive. Have a nice day!")

if __name__ == '__main__':
    main()