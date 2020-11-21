import chess
import chess.pgn
from brobot.engine import search, evaluators
import time
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
        self.t = time.time()

    def set_timeleft(self, timeleft):
        self.timeleft = timeleft
        self.t = time.time()

    def find_best_move(self):
        turn = self.board.turn
        color = 1 if turn else -1
        depth = self.depth
        nummoves = len(list(self.board.move_stack))
        if nummoves > 50:
            depth += 1
        if nummoves > 150:
            depth += 1

        (score, move) = search.negamax(self, depth, -9999, 9999, color, root=True)
        return (score, move, depth)

    def make_move(self, move):
        self.board.push(move)

    def calculate_timelimit(self):
        if not self.timeleft:
            return None
        num_moves = len(self.board.move_stack)
        time_per_move = max(1.0, (self.timeleft) / (max(10, 60.0 - num_moves)))
        return time_per_move

    def calculate_timeleft(self):
        time_per_move = self.calculate_timelimit()
        dt = (time.time() - self.t)
        return time_per_move - dt

