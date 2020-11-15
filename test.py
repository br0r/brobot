import chess
from brobot.engine import Engine, evaluators
from brobot.train.utils import get_pieces, get_train_row

fen = 'rn1qkbnr/p1pppppp/1p6/8/8/NQP5/PP1PbPPP/R1B1KBNR w KQkq - 0 4'

engine = Engine(evaluators.net_evaluator, fen=fen, color=chess.WHITE, depth=1)
engine.color = engine.board.turn

print(engine.board.turn)
move = engine.find_best_move()
print(move)

board = chess.Board()

print(get_train_row(board)[1])
