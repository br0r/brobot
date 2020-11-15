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
    piece = l.Input(shape=(32 * 5))
    square = l.Input(shape=(64*2))
    kr = tf.keras.regularizers.l1(0.01)

    generalx = l.Dense(512, activation='elu', kernel_regularizer=kr)(general)
    generalx = l.BatchNormalization()(generalx)

    piecex = l.Dense(512, activation='elu', kernel_regularizer=kr)(piece)
    piecex = l.BatchNormalization()(piecex)

    squarex = l.Dense(512, activation='elu', kernel_regularizer=kr)(square)
    squarex = l.BatchNormalization()(squarex)

    combined = l.Concatenate()([generalx, piecex, squarex])
    out = l.Dense(1024, activation='elu', kernel_regularizer=kr)(combined)
    out = l.BatchNormalization()(out)
    out = l.Dropout(0.5)(out)
    out = l.Dense(1, activation='tanh')(out)

    model = tf.keras.models.Model(inputs=[general, piece, square], outputs=out)
    #model = tf.keras.models.Model(inputs=piece, outputs=out)
    return model

if len(sys.argv) < 2:
    print('Invalid arguments (csv_data)')
    sys.exit(1)

data = open(sys.argv[1])

X = []
X2 = []
X3 = []
y = []
with open(sys.argv[1], 'r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        (fen, score) = row
        if abs(float(score)) > 2000:
            continue
        a, b, c = get_train_row(chess.Board(fen))
        X.append(a)
        X2.append(b)
        X3.append(c)
        y.append(float(score))

X = np.asarray(X).astype(np.float32)
X2 = np.asarray(X2).astype(np.float32)
X3 = np.asarray(X3).astype(np.float32)
y = np.array(y)
# Ignore mate moves
print(y)
idxtokeep = np.where(y != 9999)

X = X[idxtokeep]
X2 = X2[idxtokeep]
X3 = X3[idxtokeep]
y = y[idxtokeep]

ymax = np.max(y)
ymean = np.mean(y)
print(ymax, ymean, np.std(y))
y = (y - ymean) / ymax
print(y, np.std(y))

print(X.shape, X2.shape, y.shape)

X_train, X_test, X2_train, X2_test, X3_train, X3_test, Y_train, Y_test = train_test_split(X, X2, X3,y)
model = get_model()
model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])
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

xtrain = [
        X_train,
        X2_train,
        X3_train
        ]
xtest = [X_test, X2_test, X3_test]


#xtrain = X2_train
#xtest = X2_test

model.fit(xtrain,Y_train, epochs=30, validation_data=(xtest, Y_test),
          batch_size=128, validation_freq=1, callbacks=[early_stopping, model_checkpoint_callback])
