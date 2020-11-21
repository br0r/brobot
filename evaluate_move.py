import sys
import time
import tensorflow as tf
import chess
import numpy as np
import csv
from brobot.train.utils import get_pos_rep, get_move_rep
from brobot.engine import search

model_path = sys.argv[1]
data_path = sys.argv[2]
movequant = model_path.endswith('.tflite')

if movequant:
    model = tf.lite.Interpreter(model_path=model_path)
    model.allocate_tensors()
else:
    model = tf.keras.models.load_model(model_path)

def get_pred(pos_rep, board, moves):
    ys_= []
    gf, pf, mf, sf = pos_rep
    if movequant:
        model.set_tensor(0, [gf])
        model.set_tensor(3, [pf])
        model.set_tensor(1, [mf])
        model.set_tensor(4, [sf])

    for move in moves:
        move_rep = get_move_rep(board, move)
        if movequant:
            model.set_tensor(2, [move_rep])
            model.invoke()
            y_ = model.get_tensor(51)[0][0]
        else:
            y_ = model([np.array([gf]), np.array([pf]), np.array([mf]), np.array([sf]), np.array([move_rep])])
        ys_.append((float(y_), move))
    return ys_
    
def is_top_1(xs, x):
    return xs[0] == x

def is_top_5(xs, x):
    return x in xs[:5]

stockfish_path = 'stockfish'
engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
with open(data_path, 'r') as f:
    reader = csv.reader(f)
    top1 = 0
    top5 = 0
    ntop1 = 0
    ntop5 = 0
    total = 0.0
    ydt = 0
    naivedt = 0
    for row in reader:
        if not row:
            continue
        if len(row) == 3:
            (fen, score, ymove) = row
            board = chess.Board(fen=fen)
        else:
            (fen,) = row
            board = chess.Board(fen=fen)
            ev = engine.analyse(board, chess.engine.Limit(depth=0))
            try:
                score = float(str(ev['score'].white()))
                ymove = ev['pv'][0].uci()
            except:
                continue

        #if board.is_capture(chess.Move.from_uci(ymove)):
            #continue
        pos_rep = get_pos_rep(board)
        ys_ = []
        t = time.time()
        ys_ = get_pred(pos_rep, board, board.legal_moves)
        ys_.sort(key=lambda x: x[0], reverse=True)
        ydt += time.time() - t
        t = time.time()
        naiveys = sorted(board.legal_moves, key=lambda x: search.get_move_sort_score(board, x, True), reverse=True)
        naivedt += time.time() - t
        ys_ = [x[1].uci() for x in ys_]
        naiveys = [x.uci() for x in naiveys]
        if is_top_1(ys_, ymove):
            top1 += 1
            top5 += 1
        elif is_top_5(ys_, ymove):
            top5 += 1
        if is_top_1(naiveys, ymove):
            ntop1 += 1
            ntop5 += 1
        elif is_top_5(naiveys, ymove):
            ntop5 += 1
        total += 1
    print('naive', 'top1', ntop1 / total, 'top5', ntop5 / total, 'x/s', total / naivedt)
    print('pred', 'top1', top1 / total, 'top5', top5 / total, 'x/s', total / ydt)

engine.close()
