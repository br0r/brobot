import sys
import csv
import chess
import chess.engine
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def run(train_seq, val_seq=None, model=None):
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss='mae', metrics=['mse', 'mae'])
    print(model.summary())
    checkpoint_filepath = '/tmp/checkpoint'
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto',
        baseline=None, restore_best_weights=False
    )

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )

    model.fit(train_seq, validation_data=val_seq, epochs=100,validation_freq=1, callbacks=[early_stopping, model_checkpoint_callback])
