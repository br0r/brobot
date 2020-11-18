import time
import chess
from brobot.engine import Engine, evaluators
from brobot.train.utils import get_pieces, get_train_row

fen = 'rn1qkbnr/p1pppppp/1p6/8/8/NQP5/PP1PbPPP/R1B1KBNR w KQkq - 0 4'
fen = 'r1bqkbnr/ppppp1pp/5p2/8/2PnP1Q1/8/PP1P1PPP/RNB1KBNR w KQkq - 3 4'

# Mate in one, black to move - Rh6#
fen = '4k3/1p1n4/2p1K3/1p1n1p1r/6pp/2p5/8/5r2 b - - 9 46'
# Mate in one, white to move - Nb5#
#fen = '1rbq1bnr/ppp2Qp1/n2kp3/3pNP2/3P4/N7/PPP2PPP/R1B1KB1R w KQ - 1 9' 
# Queen take, black to move - Rxh5
#fen = 'r1bqkbnr/ppp3p1/n3p3/3pNP1Q/8/N7/PPPP1PPP/R1B1KB1R b KQkq - 0 6'
# Queen take, white to move - bxa3
#fen = 'rnb1kbnr/ppp2ppp/4p3/3p4/3P4/q4N2/PPP1PPPP/R1BQKBR1 w Qkq - 0 5'; 
# Mate in one, white to move - d4f6
#fen = '7k/p1R5/1p2Bn1p/5K2/P2B4/8/1P4PP/8 w - - 0 43' # waits until must make move
#fen = '7k/3R4/4Bn2/p4K2/p2B4/8/1P4pP/8 w - - 0 51' # must make move

#Mate in one, black to move - c6g2
#fen = 'bn2kbr1/3ppppp/2q2n2/2P3N1/8/B1P4P/P2R1PP1/4R1K1 b - - 0 21'

engine = Engine(evaluators.net_evaluator, fen=fen, color=chess.WHITE, depth=2)
engine.color = engine.board.turn

t = time.time()
print(engine.board.turn)
move = engine.find_best_move()
print(time.time() - t, move)

