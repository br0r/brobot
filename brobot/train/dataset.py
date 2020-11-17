from multiprocessing.pool import Pool
from functools import partial
import csv
import numpy as np
import tensorflow as tf
import chess
from brobot.train.utils import get_train_row
AUTOTUNE = tf.data.experimental.AUTOTUNE


def worker(row):
    (fen, score) = row
    board = chess.Board(fen)
    a, b, c, = get_train_row(board)
    return a, b, c, float(score)

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def serialize(a, b, c, y):
    feature = {
        'a': _float_feature(a),
        'b': _float_feature(b),
        'c': _float_feature(c),
        'y': _float_feature([y])
    }
    proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return proto.SerializeToString()

def createRecord(csv_file_path, filename):
    #pool = Pool(1)
    i = 0
    with tf.io.TFRecordWriter(filename) as writer:
        with open(csv_file_path, 'r') as csv_file:
            reader = csv.reader(csv_file)
            batch = []
            for row in reader:
                a, b, c, y = worker(row)
                example = serialize(a, b, c, y)
                writer.write(example)
                i += 1
                if i % 100000 == 0:
                    print(i)
                """
                batch.append(row)
                if len(batch) > BATCH_SIZE:
                    data = pool.map(worker, batch)
                    print('got data', len(data))
                    for (a, b, c, y) in data:
                        example = serialize(a, b, c, y)
                        writer.write(example)
                    batch = []
                """

def read_tfrecord(example):
    tfrecord_format = (
        {
            "a": tf.io.FixedLenFeature([], tf.float32),
            "b": tf.io.FixedLenFeature([], tf.float32),
            "c": tf.io.FixedLenFeature([], tf.float32),
            "y": tf.io.FixedLenFeature([], tf.float32),
        }
    )
    example = tf.io.parse_single_example(example, tfrecord_format)
    print('EXAMPLE', example)
    a = example["a"]
    b = example["b"]
    c = example["c"]
    y = example["y"]
    return a, b, c, y

def load_dataset(filenames, labeled=True):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(
        filenames,
    )  # automatically interleaves reads from multiple files
    dataset.output_types = ({'a': tf.float64, 'b': tf.float64, 'c': tf.float64}, tf.float64)
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(
        partial(read_tfrecord), num_parallel_calls=AUTOTUNE
    )
    # returns a dataset of (image, label) pairs if labeled=True or just images if labeled=False
    return dataset


def create_dataset(filenames, BATCH_SIZE):
    def gen():
        X = []
        X2 = []
        X3 = []
        Y = []
        dataset = tf.data.TFRecordDataset(
            filenames,
        )  # automatically interleaves reads from multiple files
        for record in dataset:
            #a, b, c, y = read_tfrecord(record)
            example = tf.train.Example()
            example.ParseFromString(record.numpy())
            #print(dir(example.features.feature))
            a = example.features.feature.get('a').float_list.value
            b = example.features.feature.get('b').float_list.value
            c = example.features.feature.get('c').float_list.value
            y = example.features.feature.get('y').float_list.value
            #y = y/2000
            X.append(a)
            X2.append(b)
            X3.append(c)
            Y.append(y)
            if len(Y) > BATCH_SIZE:
                #print(np.array(X).astype(np.float64))
                yield {'a': np.array(X).astype(np.float32), 'b': np.array(X2).astype(np.float32), 'c': np.array(X3).astype(np.float32)}, np.array(Y).astype(np.float32)
                X = []
                X2 = []
                X3 = []
                Y = []

    dataset = tf.data.Dataset.from_generator(gen, output_types=({'a': tf.float64, 'b': tf.float64, 'c': tf.float64}, tf.float64))
    return dataset
    #return gen()
