import chess
from brobot.engine import Engine, evaluators
fen = '3k4/5RQ1/8/8/2N1B3/8/P1P2P1P/1K2R3 w - - 19 42'

engine = Engine(evaluators.simple_evaluator, fen=fen, color=chess.WHITE, depth=3)
engine.color = engine.board.turn

print(engine.board.turn)
move = engine.find_best_move()
print(move)
