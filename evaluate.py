import sys
import time
import tensorflow as tf
import numpy as np
from brobot.train.dataset import SerializedSequence

model_path = sys.argv[1]
data_path = sys.argv[2]

s = SerializedSequence(data_path, mem=True, multi=True)

quant = model_path.endswith('.tflite')
if quant:
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(output_details)
    errors = []
    t = time.time()
    for row in s.data:
        x, y = row
        general, piece, mobility, square = x
        # Test the model on random input data.
        interpreter.set_tensor(0, [general])
        interpreter.set_tensor(1, [mobility])
        interpreter.set_tensor(2, [piece])
        interpreter.set_tensor(3, [square])

        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        y_ = output_data[0][0]
        e = abs((y / 100) - y_)
        errors.append(e)
    print('mae', np.mean(errors))
    print('speed',  len(s.data) / (time.time() - t))
else:
    model = tf.keras.models.load_model(model_path)
    errors = []
    t = time.time()
    for row in s.data:
        x, y = row
        gf, pf, mf, sf = x
        y_ = model([np.array([gf]), np.array([pf]), np.array([mf]), np.array([sf])])
        e = abs((y / 100) - y_)
        errors.append(e)

    print('mae', np.mean(errors))
    print('speed',  len(s.data) / (time.time() - t))
    evaluate = model.evaluate(s)
    print(evaluate)
