import sys
import csv
import chess
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils import get_train_row
import pickle

def get_model():
    l = tf.keras.layers
    general = l.Input(shape=(15,))
    piece = l.Input(shape=(32 * 5))
    square = l.Input(shape=(64*2))
    #kr = tf.keras.regularizers.l1(0.01)
    kr = None

    n = 128

    generalx = l.Dense(32, activation='relu', kernel_regularizer=kr)(general)
    #generalx = l.BatchNormalization()(generalx)

    piecex = l.Dense(512, activation='relu', kernel_regularizer=kr)(piece)
    #piecex = l.BatchNormalization()(piecex)

    squarex = l.Dense(256, activation='relu', kernel_regularizer=kr)(square)
    #squarex = l.BatchNormalization()(squarex)

    combined = l.Concatenate()([generalx, piecex, squarex])
    #combined = l.Concatenate()([generalx, piecex])
    out = l.Dense(512, activation='relu', kernel_regularizer=kr)(combined)
    #out = l.BatchNormalization()(out)
    out = l.Dropout(0.5)(out)
    out = l.Dense(1, activation='linear')(out)

    model = tf.keras.models.Model(inputs=[general, piece, square], outputs=out)
    #model = tf.keras.models.Model(inputs=[general, piece], outputs=out)
    #model = tf.keras.models.Model(inputs=piece, outputs=out)
    return model

if len(sys.argv) < 2:
    print('Invalid arguments (csv_data)')
    sys.exit(1)


def gen():
    X = []
    X2 = []
    X3 = []
    y = []
    with open(sys.argv[1], 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            (fen, score) = row
            print(fen)
            if abs(float(score)) > 2000:
                continue
            a, b, c = get_train_row(chess.Board(fen))

            yield {'a': a, 'b': b, 'c': c}, float(score) / 2000

model = get_model()
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss='mae', metrics=['mse', 'mae'])
print(model.summary())
checkpoint_filepath = '/tmp/checkpoint'
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto',
    baseline=None, restore_best_weights=False
)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True
)

dataset = tf.data.Dataset.from_generator(gen, output_types=({'a': tf.float64, 'b': tf.float64, 'c': tf.float64}, tf.float64))

#xtrain = X2_train
#xtest = X2_test

model.fit(dataset, epochs=100, batch_size=128, validation_freq=1, callbacks=[early_stopping, model_checkpoint_callback])
