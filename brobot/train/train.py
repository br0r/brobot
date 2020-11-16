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
    general = l.Input(shape=(16,))
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


def create_dataset(csv_file_path, BATCH_SIZE):
    def gen():
        X = []
        X2 = []
        X3 = []
        Y = []
        with open(csv_file_path, 'r') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                (fen, score) = row
                y = float(score)
                #if abs(y) > 2000:
                    #continue
                a, b, c = get_train_row(chess.Board(fen))

                #y = y/2000
                X.append(a)
                X2.append(b)
                X3.append(c)
                Y.append(y)
                if len(Y) > BATCH_SIZE:
                    yield {'a': np.array(X).astype(np.float32), 'b': np.array(X2).astype(np.float32), 'c': np.array(X3).astype(np.float32)}, np.array(Y).astype(np.float32)
                    X = []
                    X2 = []
                    X3 = []
                    Y = []
    dataset = tf.data.Dataset.from_generator(gen, output_types=({'a': tf.float64, 'b': tf.float64, 'c': tf.float64}, tf.float64))
    return dataset

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



#dataset = tf.data.Dataset.from_generator(gen, output_shapes=((tf.TensorShape((15,)), tf.SparseTensorSpec((32*5,))), tf.TensorSpec((64*2)), tf.TensorSpec(())))


#xtrain = X2_train
#xtest = X2_test

model.fit(create_dataset(sys.argv[1], 128), validation_data=create_dataset(sys.argv[2], 128), epochs=100,validation_freq=1, callbacks=[early_stopping, model_checkpoint_callback])
