import chess
import chess.pgn
from brobot.engine import search, evaluators
import numpy as np

class Engine:
    def __init__(self, evaluator, fen=False, name="", color=True, depth=3, totaltime=None):
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

    def set_timeleft(self, timeleft):
        self.timeleft = timeleft

    def find_best_move(self):
        turn = self.board.turn
        timelimit = self.calculate_timelimit()
        return search.iterative_deepening(self.board, self.depth, self.color != turn, self.evaluator, timelimit, self.transition_table)

    def make_move(self, move):
        self.board.push(move)

    def calculate_timelimit(self):
        if not self.timeleft:
            return None
        num_moves = len(self.board.move_stack)
        return max(1.0, self.timeleft / (max(10, 60.0 - num_moves)))

