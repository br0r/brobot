import sys
import csv
import chess
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils import get_train_row

def get_model():
    l = tf.keras.layers
    general = l.Input(shape=(15,))
    piece = l.Input(shape=(32 * 3))


    generalx = l.Dense(2048, activation='relu')(general)

    piecex = l.Dense(2048, activation='relu')(piece)

    combined = l.Concatenate()([generalx, piecex])
    out = l.Dense(2048, activation='relu')(combined)
    out = l.Dense(1, activation='tanh')(out)

    model = tf.keras.models.Model(inputs=[general, piece], outputs=out)
    return model

if len(sys.argv) < 2:
    print('Invalid arguments (csv_data)')
    sys.exit(1)

data = open(sys.argv[1])

X = []
X2 = []
y = []
with open(sys.argv[1], 'r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        (fen, score) = row
        a, b = get_train_row(chess.Board(fen))
        X.append(a)
        X2.append(b)
        y.append(float(score))

X = np.asarray(X).astype(np.float32)
X2 = np.asarray(X2).astype(np.float32)
y = np.array(y)
ymax = np.max(y)
y /= ymax

print(X.shape, X2.shape, y.shape)

X_train, X_test, X2_train, X2_test, Y_train, Y_test = train_test_split(X, X2,y)
optimizer = tf.keras.optimizers.SGD(
    learning_rate=0.001, momentum=0.7, nesterov=True
    #learning_rate=0.01, momentum=0.7, nesterov=True
)
model = get_model()
model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])
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

xtrain = [
        X_train,
        X2_train
        ]
xtest = [X_test, X2_test]


model.fit(xtrain,Y_train, epochs=100, validation_data=(xtest, Y_test),
          batch_size=256, validation_freq=1, callbacks=[early_stopping, model_checkpoint_callback])

