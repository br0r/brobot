import tensorflow as tf
import numpy as np
import time
from chess.polyglot import zobrist_hash
from brobot.train.utils import get_pos_rep, get_move_rep

gmodel = None
gmovemodel = None
#WEIGHTS_PATH = '/tmp/checkpoint'
#WEIGHTS_PATH = '/Users/bror/workspace/workshops/checkpoint'
WEIGHTS_PATH = '/Users/bror/workspace/workshops/mk2-3.tflite'
MOVE_MODEL_PATH = '/Users/bror/workspace/workshops/move-mk1-3.tflite'

quant = WEIGHTS_PATH.endswith('.tflite')
movequant = MOVE_MODEL_PATH.endswith('.tflite')
cache = {}
movecache = {}
# negamax impl
def net_evaluator(board):
    turn = board.turn
    mul = 1 if turn else -1
    if board.is_checkmate():
        return -mul * (9000 - len(board.move_stack))
    if board.is_stalemate():
        return 0

    global gmodel
    if not gmodel:
        if quant:
            gmodel = tf.lite.Interpreter(model_path=WEIGHTS_PATH)
            gmodel.allocate_tensors()
        else:
            gmodel = tf.keras.models.load_model(WEIGHTS_PATH)
    h = zobrist_hash(board)
    if h in cache:
        return cache[h]

    gf, pf, mf, sf = get_pos_rep(board)
    if quant:
        gmodel.set_tensor(0, [gf])
        gmodel.set_tensor(1, [pf])
        gmodel.set_tensor(3, [mf])
        gmodel.set_tensor(2, [sf])
        gmodel.invoke()
        score = gmodel.get_tensor(42)[0][0]
    else:
        score = gmodel([np.array([gf]), np.array([pf]), np.array([mf]), np.array([sf])])
    score = float(score)
    cache[h] = score
    return score

def get_moves_pred(board, moves, h=None):
    if not h:
        h = zobrist_hash(board)
    if h in movecache:
        return movecache[h]
    global gmovemodel
    if not gmovemodel:
        if movequant:
            gmovemodel = tf.lite.Interpreter(model_path=MOVE_MODEL_PATH)
            gmovemodel.allocate_tensors()
        else:
            gmovemodel = tf.keras.models.load_model(MOVE_MODEL_PATH)
    ys_= []
    gf, pf, mf, sf = get_pos_rep(board)
    if movequant:
        gmovemodel.set_tensor(0, [gf])
        gmovemodel.set_tensor(3, [pf])
        gmovemodel.set_tensor(1, [mf])
        gmovemodel.set_tensor(4, [sf])

    for move in moves:
        move_rep = get_move_rep(board, move)
        if movequant:
            gmovemodel.set_tensor(2, [move_rep])
            gmovemodel.invoke()
            y_ = gmovemodel.get_tensor(51)[0][0]
        else:
            y_ = gmovemodel([np.array([gf]), np.array([pf]), np.array([mf]), np.array([sf]), np.array([move_rep])])
        ys_.append((float(y_), move))
    movecache[h] = ys_
    return ys_
