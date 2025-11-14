import random

class RandomAgent:
    def __init__(self, color):
        self.color = color  # 'w' or 'b'

    def select_move(self, game):
        """
        Given a Hive game instance, select a random valid move for this agent's color.
        Prints the first 5 valid moves and the total number of possible moves.
        """
        valid_moves = game.find_valid_moves(self.color)
        #print(f"Agent ({self.color}) has {len(valid_moves)} possible moves.")
        #for i, move in enumerate(valid_moves[:5]):
            #print(f"Move {i+1}: {move}")
        if not valid_moves:
            return ('non_play', 'pass')
        move = random.choice(valid_moves)
        #print(f"\nRandomAgent ({self.color}) chose move: {move}\n")
        return move