import os
import sys
import tensorflow as tf
import chess
import chess.engine
from brobot.train.dataset import SerializedSequence, build_serialized_data, parse_fen_row, parse_tuner
from brobot.train.train import run
from brobot.train.models import get_old_model
from brobot.train.utils import get_train_row_old


def create_serialize():
    STOCKFISH_PATH = os.getenv('STOCKFISH_PATH', 'stockfish')
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    build_serialized_data(sys.argv[1], './train.file', parse_tuner(get_train_row_old, engine=engine))
    build_serialized_data(sys.argv[2], './val.file', parse_fen_row(get_train_row_old))
    engine.close()

#create_serialize()

#WEIGHTS_PATH = '/tmp/checkpoint'
#model = tf.keras.models.load_model(WEIGHTS_PATH)
model = get_old_model()
mem = True
train_seq = SerializedSequence('./train.file', 128, mem=mem)
val_seq = SerializedSequence('./val.file', 128, mem=mem)
run(train_seq, val_seq, model=model)

