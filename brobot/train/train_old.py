import sys
import csv
import chess
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils import get_train_row_old

def get_model():
    l = tf.keras.layers
    model = tf.keras.models.Sequential([
        l.Input(shape=(768,)),
        l.Dense(2048, activation='elu'),
        l.BatchNormalization(),
        l.Dense(2048, activation='elu'),
        l.BatchNormalization(),
        l.Dense(2048, activation='elu'),
        #l.Dropout(0.5),
        l.Dense(1, activation='linear'),
    ])
    return model


if len(sys.argv) < 2:
    print('Invalid arguments (csv_data)')
    sys.exit(1)

data = open(sys.argv[1])

X = []
y = []
with open(sys.argv[1], 'r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        (fen, score) = row
        if abs(float(score)) > 2000:
            continue
        a = get_train_row_old(chess.Board(fen))
        X.append(a)
        y.append(float(score))

X = np.asarray(X).astype(np.float32)
y = np.array(y)
idxtokeep = np.where(y != 9999)
X = X[idxtokeep]
y = y[idxtokeep]
ymax = np.max(y)
ymean = np.mean(y)
print(ymax, ymean, np.std(y))
#y = (y - ymean) / ymax
#y = (y - ymean) / ymax
y = y / ymax
print(y, np.std(y))

print(X.shape, y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, y)

optimizer = tf.keras.optimizers.SGD(
    learning_rate=0.001, momentum=0.7, nesterov=True
)

model = get_model()
model.compile(optimizer=optimizer, loss='mse', metrics=['mse', 'mae'])
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

xtrain = X_train
xtest = X_test

model.fit(xtrain,Y_train, epochs=100, validation_data=(xtest, Y_test),
          batch_size=256, validation_freq=1, callbacks=[early_stopping, model_checkpoint_callback])

