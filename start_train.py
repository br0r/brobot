import os
import sys
import tensorflow as tf
from brobot.train.dataset import SerializedSequence
from brobot.train.train import run
from brobot.train.models import get_old_model
from brobot.train.utils import get_train_row_old

#WEIGHTS_PATH = '/tmp/checkpoint'
#model = tf.keras.models.load_model(WEIGHTS_PATH)
model = get_old_model()
mem = True
train_seq = SerializedSequence('./train.file', 128, mem=mem)
val_seq = SerializedSequence('./val.file', 128, mem=mem)
run(train_seq, val_seq, model=model)

