import chess
import chess.pgn
from brobot.engine import search, evaluators
import numpy as np

class Engine:
    def __init__(self, evaluator, fen=False, name="", color=True, depth=1, totaltime=None):
        self.name = name
        self.color = color
        if fen:
            self.board = chess.Board(fen)
        else:
            self.board = chess.Board()
        self.evaluator = evaluator
        self.depth = depth
        self.timeleft = totaltime
        self.transition_table = {}
        self.transition_table = {}

    def set_timeleft(self, timeleft):
        self.timeleft = timeleft

    def find_best_move(self):
        turn = self.board.turn
        color = 1 if turn else -1
        (score, move) = search.negamax(self, self.depth, -9999, 9999, color)
        return (score, move)

    def make_move(self, move):
        self.board.push(move)

    def calculate_timelimit(self):
        if not self.timeleft:
            return None
        num_moves = len(self.board.move_stack)
        return max(1.0, self.timeleft / (max(10, 60.0 - num_moves)))

