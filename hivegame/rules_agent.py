import random

class RulesAgent:
    def __init__(self, color):
        self.color = color

    def select_move(self, game):
        valid_moves = game.find_valid_moves(self.color)
        # 1. Win if possible
        for move in valid_moves:
            if self._move_surrounds_enemy_queen(game, move):
                return move
        # 2. Block if opponent can win next turn (not implemented here)
        # 3. Place Queen if not placed and must be placed soon
        if not self._queen_placed(game):
            for move in valid_moves:
                if move[0] == 'play' and move[1][0] == f'{self.color}Q1':
                    return move
        # 4. Place next to opponent Queen if possible
        for move in valid_moves:
            if self._move_next_to_enemy_queen(game, move):
                return move
        # 5. Otherwise, pick a random move
        return random.choice(valid_moves) if valid_moves else ('non_play', 'pass')

    def _queen_placed(self, game):
        return f'{self.color}Q1' in game.playedPieces

    def _move_surrounds_enemy_queen(self, game, move):
        """
        Returns True if this move would result in the enemy queen being surrounded (i.e., a win).
        """
        # Determine enemy color and queen name
        enemy_color = 'b' if self.color == 'w' else 'w'
        enemy_queen = f'{enemy_color}Q1'
        # If enemy queen is not on the board, can't surround
        if enemy_queen not in game.playedPieces:
            return False
        queen_cell = game.playedPieces[enemy_queen]['cell']
        # Count how many neighbors are occupied now
        occupied_now = set(game._occupied_surroundings(queen_cell))
        # Simulate the move: does it add a neighbor to the queen?
        if move[0] == 'play':
            piece_name, ref_piece, direction = move[1]
            if ref_piece is None or direction is None:
                return False
            target_cell = game._poc2cell(ref_piece, direction)
        elif move[0] == 'move':
            piece_name, ref_piece, direction = move[1]
            target_cell = game._poc2cell(ref_piece, direction)
        else:
            return False
        # If the move places a piece next to the queen, add that cell
        if target_cell in game.board.get_surrounding(queen_cell):
            occupied_now = occupied_now | {target_cell}
        # If after this move, all 6 neighbors are occupied, it's a win
        return len(occupied_now) == 6

    def _move_next_to_enemy_queen(self, game, move):
        """
        Returns True if this move would place a piece adjacent to the enemy queen.
        """
        enemy_color = 'b' if self.color == 'w' else 'w'
        enemy_queen = f'{enemy_color}Q1'
        if enemy_queen not in game.playedPieces:
            return False
        queen_cell = game.playedPieces[enemy_queen]['cell']
        if move[0] == 'play':
            piece_name, ref_piece, direction = move[1]
            if ref_piece is None or direction is None:
                return False
            target_cell = game._poc2cell(ref_piece, direction)
        elif move[0] == 'move':
            piece_name, ref_piece, direction = move[1]
            target_cell = game._poc2cell(ref_piece, direction)
        else:
            return False
        # Is the move adjacent to the enemy queen?
        return target_cell in game.board.get_surrounding(queen_cell)