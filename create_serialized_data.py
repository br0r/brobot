import os
import sys
import chess
import chess.engine
from brobot.train.dataset import build_serialized_data, parse_fen_row, parse_tuner
from brobot.train.utils import get_train_row_old, get_train_row

STOCKFISH_PATH = os.getenv('STOCKFISH_PATH', 'stockfish')
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

#build_serialized_data(sys.argv[1], sys.argv[2], parse_tuner(get_train_row_old, engine=engine, depth=0), verbose=True)
build_serialized_data(sys.argv[1], sys.argv[2], parse_tuner(get_train_row, engine=engine, depth=0), verbose=True)
#build_serialized_data(sys.argv[1], sys.argv[2], parse_fen_row(get_train_row), verbose=True)
engine.close()
