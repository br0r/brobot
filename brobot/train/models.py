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
        l.BatchNormalization(),
        l.Dropout(0.5),
        l.Dense(1, activation='linear'),
    ])
    return model

def get_model():
    l = tf.keras.layers
    general = l.Input(shape=(21,), name='a')
    piece = l.Input(shape=(32 * 7), name='b')
    mobility = l.Input(shape=(4 * 10), name='mobility')
    square = l.Input(shape=(64*4), name='c')
    #kr = tf.keras.regularizers.l1(0.01)
    kr = None
    # best 1.47 s = 3, dropout all, quiet data
    s = 2
    generalx = l.Dense(32 * s, activation='relu', kernel_regularizer=kr)(general)
    generalx = l.BatchNormalization()(generalx)
    generalx = l.Dropout(0.5)(generalx)

    piecex = l.Dense(512 * s, activation='relu', kernel_regularizer=kr)(piece)
    piecex = l.BatchNormalization()(piecex)
    piecex = l.Dropout(0.5)(piecex)

    mobilityx = l.Dense(128 * s, activation='relu', kernel_regularizer=kr)(mobility)
    mobilityx = l.BatchNormalization()(mobilityx)
    mobilityx = l.Dropout(0.5)(mobilityx)

    squarex = l.Dense(256 * s, activation='relu', kernel_regularizer=kr)(square)
    squarex = l.BatchNormalization()(squarex)
    squarex = l.Dropout(0.5)(squarex)

    combined = l.Concatenate()([generalx, piecex, mobilityx, squarex])
    out = l.Dense(512 * s, activation='relu', kernel_regularizer=kr)(combined)
    out = l.BatchNormalization()(out)
    out = l.Dropout(0.5)(out)
    out = l.Dense(1, activation='linear')(out)

    model = tf.keras.models.Model(inputs=[general, piece, mobility, square], outputs=out)
    return model
