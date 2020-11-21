import os
import sys
import tensorflow as tf
from brobot.train.dataset import SerializedSequence
from brobot.train.train import run
from brobot.train.models import get_move_model

val_seq = None
#WEIGHTS_PATH = '/tmp/checkpoint'
#model = tf.keras.models.load_model(WEIGHTS_PATH)
#model = get_old_model()
mem = True
multi = True
train_seq = SerializedSequence('./movetest.file', 128, mem=mem, multi=multi)
val_seq = SerializedSequence('./movetest.val.file', 128, mem=mem, multi=multi)
model = get_move_model()
run(train_seq, val_seq=val_seq, model=model, checkpoint_filepath='/tmp/checkpoint_move', loss='mse')

