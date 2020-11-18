import os
import sys
from brobot.train.dataset import parse_tuner, build_serialized_data, SerializedSequence
from brobot.train.utils import get_train_row_old
import chess
import chess.engine
import csv



s = SerializedSequence('./tuner.old.file', 128)

csvpath = sys.argv[1]

#STOCKFISH_PATH = os.getenv('STOCKFISH_PATH', 'stockfish')
#engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
#build_serialized_data(csvpath, './tuner.old.file', parse_tuner(get_train_row_old, engine=engine))
#engine.close()
