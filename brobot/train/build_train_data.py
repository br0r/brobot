import os
import sys
import chess
import chess.pgn
import chess.engine
import json
import csv

if len(sys.argv) < 4:
    print('Invalid arguments (pgn, stockfish, n)')
    sys.exit(1)
pgn_file = sys.argv[1]
stockfish_path = sys.argv[2]
n = sys.argv[3]

if not pgn_file:
    print('No pgn')
    sys.exit(1)

if not stockfish_path:
    print('No stockfish path')
    sys.exit(1)

count = 0
engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
writer = csv.writer(sys.stdout)
with open(pgn_file, 'r') as pgn:
    for i in range(int(n)):
        game = chess.pgn.read_game(pgn)
        board = chess.Board()
        for move in game.mainline_moves():
            board.push(move)
            ev = engine.analyse(board, chess.engine.Limit(depth=0))
            fen = board.fen()
            score = ev['score'].white()
            if isinstance(score, chess.engine.Mate) or isinstance(score, chess.engine.MateGivenType):
                y = 9999
            else:
                y = float(str(score))
            writer.writerow([fen, y])

pgn.close()
engine.close()
