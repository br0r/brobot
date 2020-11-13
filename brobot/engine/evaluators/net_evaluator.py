import tensorflow as tf
import numpy as np
from brobot.train.utils import get_train_row
import time
gmodel = None
WEIGHTS_PATH = '/tmp/checkpoint'
def net_evaluator(board):
    global gmodel
    if not gmodel:
        gmodel = tf.keras.models.load_model(WEIGHTS_PATH)
    s = time.time()
    row = get_train_row(board)
    print('get', time.time() - s)
    s = time.time()
    score = gmodel.predict(np.array([row]))[0]
    print('pred', time.time() - s)
    return score
