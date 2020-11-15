import tensorflow as tf
import numpy as np
import time
from chess.polyglot import zobrist_hash
from brobot.train.utils import get_train_row, get_train_row_old

gmodel = None
WEIGHTS_PATH = '/tmp/checkpoint'
cache = {}
def net_evaluator(board):
    if board.is_checkmate():
        if board.turn:
            return -9999
        else:
            return 9999

    if board.is_stalemate():
        return 0
    if board.is_insufficient_material():
        return 0

    global gmodel
    if not gmodel:
        gmodel = tf.keras.models.load_model(WEIGHTS_PATH)
    h = zobrist_hash(board)
    if h in cache:
        if board.turn:
            return cache[h]
        else:
            return -cache[h]

    #gf, pf = get_train_row(board)
    #score = gmodel.predict([np.array([gf]), np.array([pf])])[0]

    x = get_train_row_old(board)
    score = gmodel.predict(np.array([x]))[0]

    #gf, pf = get_train_row(board)
    #score = gmodel.predict(np.array([pf]))[0]
    cache[h] = score
    if board.turn:
        return score
    else:
        return -score
