import os
import sys
import tensorflow as tf
from brobot.train.dataset import SerializedSequence
from brobot.train.train import run
from brobot.train.models import get_old_model, get_model

val_seq = None
#WEIGHTS_PATH = '/tmp/checkpoint'
#model = tf.keras.models.load_model(WEIGHTS_PATH)
#model = get_old_model()
mem = True
multi = True
train_seq = SerializedSequence('./test.file', 128, mem=mem, multi=multi)
#val_seq = SerializedSequence('./val.file', 128, mem=mem, multi=multi)
model = get_model()
run(train_seq, val_seq=val_seq, model=model)

