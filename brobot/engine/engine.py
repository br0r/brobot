import chess
import chess.pgn
from brobot.engine import search, evaluators
import time
import math
import numpy as np

PROB_THRESHOLD = 0.0000001
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
        self.prob_threshold = PROB_THRESHOLD * 10

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
            self.prob_threshold = PROB_THRESHOLD / 10
        if nummoves > 150:
            depth += 1
            self.prob_threshold = PROB_THRESHOLD / 100
            depth += math.floor((nummoves - 150) / 50)

        (score, move, node_depth) = search.negamax(self, depth, -9999, 9999, color, root=True, prob=1.0, curr_depth=0)
        return (score, move, node_depth)

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

