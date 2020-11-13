import tensorflow as tf
import numpy as np
import time
from chess.polyglot import zobrist_hash
from brobot.train.utils import get_train_row

gmodel = None
WEIGHTS_PATH = '/tmp/checkpoint'
cache = {}
def net_evaluator(board):
    global gmodel
    if not gmodel:
        gmodel = tf.keras.models.load_model(WEIGHTS_PATH)
    h = zobrist_hash(board)
    if h in cache:
        return cache[h]

    gf, pf = get_train_row(board)
    score = gmodel.predict([np.array([gf]), np.array([pf])])[0]
    cache[h] = score
    return score
