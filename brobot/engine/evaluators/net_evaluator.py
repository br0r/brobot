import tensorflow as tf
import numpy as np
import time
from chess.polyglot import zobrist_hash
from brobot.train.utils import get_train_row, get_train_row_old

gmodel = None
#WEIGHTS_PATH = '/tmp/checkpoint'
WEIGHTS_PATH = '/Users/bror/workspace/workshops/checkpoint'
cache = {}
# negamax impl
def net_evaluator(board):
    turn = board.turn
    mul = 1 if turn else -1
    if board.is_checkmate():
        return -mul * (9000 - len(board.move_stack))

    global gmodel
    if not gmodel:
        gmodel = tf.keras.models.load_model(WEIGHTS_PATH)
    h = zobrist_hash(board)
    if h in cache:
        # return mul * cache[h]
        return cache[h]

    gf, pf, mf, sf = get_train_row(board)
    score = gmodel([np.array([gf]), np.array([pf]), np.array([mf]), np.array([sf])])
    score = float(score)
    #score = gmodel.predict([np.array([gf]), np.array([pf]), np.array([sf])])[0]
    #score = gmodel.predict([np.array([gf]), np.array([pf])])[0]

    #x = get_train_row_old(board)
    #score = gmodel.predict(np.array([x]))[0]

    #gf, pf = get_train_row(board)
    #score = gmodel.predict(np.array([pf]))[0]
    cache[h] = score
    #return mul * score
    return score
