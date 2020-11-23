import os
import sys
import chess
import chess.engine
from brobot.train.dataset import build_serialized_data, parse_fen_row, parse_tuner, parse_fen_row_move, parse_tuner_move, build_serialized_data_from_bin, parse_fen_row_bin
from brobot.train.utils import get_pos_rep, get_move_rep

STOCKFISH_PATH = os.getenv('STOCKFISH_PATH', 'stockfish')
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

#build_serialized_data(sys.argv[1], sys.argv[2], parse_tuner(get_pos_rep, engine=engine, depth=0), verbose=True)
#build_serialized_data(sys.argv[1], sys.argv[2], parse_fen_row(get_pos_rep), verbose=True)
#build_serialized_data(sys.argv[1], sys.argv[2], parse_fen_row_move(get_pos_rep, get_move_rep), verbose=True)
#build_serialized_data(sys.argv[1], sys.argv[2], parse_tuner_move(get_pos_rep, get_move_rep, engine=engine, depth=0), verbose=True)
build_serialized_data_from_bin(sys.argv[1], sys.argv[2], parse_fen_row_bin(get_pos_rep), verbose=True, limit=1e6)

engine.close()
