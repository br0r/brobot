import tensorflow as tf

def get_old_model():
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

def get_model():
    l = tf.keras.layers
    general = l.Input(shape=(16,), name='a')
    piece = l.Input(shape=(32 * 5), name='b')
    square = l.Input(shape=(64*2), name='c')
    #kr = tf.keras.regularizers.l1(0.01)
    kr = None

    generalx = l.Dense(32, activation='relu', kernel_regularizer=kr)(general)
    generalx = l.BatchNormalization()(generalx)

    piecex = l.Dense(512, activation='relu', kernel_regularizer=kr)(piece)
    piecex = l.BatchNormalization()(piecex)

    squarex = l.Dense(256, activation='relu', kernel_regularizer=kr)(square)
    squarex = l.BatchNormalization()(squarex)

    combined = l.Concatenate()([generalx, piecex, squarex])
    out = l.Dense(512, activation='relu', kernel_regularizer=kr)(combined)
    out = l.BatchNormalization()(out)
    out = l.Dropout(0.5)(out)
    out = l.Dense(1, activation='linear')(out)

    model = tf.keras.models.Model(inputs=[general, piece, square], outputs=out)
    return model
