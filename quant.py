import sys
import tensorflow as tf
saved_model_dir = sys.argv[1]
quant_model_path = sys.argv[2]
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
tflite_quant_model = converter.convert()

with open(quant_model_path, 'wb') as f:
    f.write(tflite_quant_model)
