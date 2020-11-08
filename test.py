from engine import Engine
import evaluators
from utils import eprint
fen = 'r2qr1k1/ppp2ppp/4bn2/3p4/8/7n/P3K3/8 b - - 1 23'

engine = Engine(evaluators.simple_evaluator, fen=fen, color=chess.BLACK)

print(engine.board.turn)
move = engine.find_best_move()
eprint(move)
