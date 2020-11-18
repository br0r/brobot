import sys
from functools import partial
import csv
import numpy as np
import tensorflow as tf
import chess
import chess.engine
from brobot.train.utils import get_train_row

def parse_fen_row(parsef):
    def parser(row):
        (fen, score) = row
        board = chess.Board(fen=fen)
        x = parsef(board)
        return x, float(score)
    return parser

def parse_tuner(parsef, engine, depth=4):
    def parser(row):
        (fen,) = row
        board = chess.Board(fen=fen)
        ev = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = ev['score'].white()
        if isinstance(score, chess.engine.Mate) or isinstance(score, chess.engine.MateGivenType):
            y = None # Ignore mates
        else:
            y = float(str(score))

        if not y:
            return None

        x = parsef(board)
        return x, y
    return parser

def build_serialized_data(csv_file, to, parsef, verbose=False):
    if not csv_file or not to:
        raise 'Invalid arguments'
    open(to, 'w').close()
    with open(to, 'ab') as writef:
        with open(csv_file, 'r') as readf:
            reader = csv.reader(readf)
            i = 0
            for row in reader:
                if not row:
                    continue
                arr = parsef(row)
                if not arr:
                    continue
                #np.savetxt(writef, arr, fmt='%d')
                np.save(writef, arr)
                i += 1
                if i % 100 == 0 and verbose:
                    sys.stdout.write('\r%d' % i)
                    sys.stdout.flush()

class SerializedSequence(tf.keras.utils.Sequence):
    def __init__(self, sequence_file_path, batch_size, mem=False, multi=False):
        self.batch_size = batch_size
        self.sequence_file_path = sequence_file_path
        i = 0
        self.chunks = []
        self.file = open(self.sequence_file_path, 'rb')
        self.multi = multi
        self.mem = mem
        self.data = []

        #s = 0
        while True:
            try:
                arr = np.load(self.file, allow_pickle=True)
                #e = self.file.tell()
                #self.chucks.append((s, e))
                if self.mem:
                    self.data.append(arr)
                i += 1
            except:
                break

        self.file.seek(0)
        self.len = i
        print('LEN', self.len)

    def __len__(self):
        return self.len // self.batch_size

    def on_epoch_end(self):
        if self.mem:
            np.random.shuffle(self.data)
        else:
            self.file.seek(0)

    def __getitem__(self, idx):
        if self.mem:
            batch = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        else:
            batch = []
            while True:
                try:
                    arr = np.load(self.file, allow_pickle=True)
                    batch.append(list(arr))
                    if len(batch) >= self.batch_size:
                        break
                except:
                    break
        x,y = zip(*batch)
        if self.multi:
            xs = [np.array(a) for a in zip(*x)]
            ys = [np.array(y) / 100]
            return xs, ys
        else:
            return np.array(x), np.array(y) / 100
