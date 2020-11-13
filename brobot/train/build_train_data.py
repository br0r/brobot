import os
import sys
import chess
import chess.pgn
import chess.engine
import json
import csv
from utils import eprint

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

engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
writer = csv.writer(sys.stdout)
with open(pgn_file, 'r') as pgn:
    count = 0
    while count < int(n):
        game = chess.pgn.read_game(pgn)
        if not game:
            eprint('Consumed')
            break
        board = chess.Board()
        try:
            headers = game.headers
            belo = headers.get('BlackElo', 0)
            belo = 0 if belo == '?' else int(belo)
            welo = int(headers.get('WhiteElo', 0))
            welo = 0 if welo == '?' else int(welo)
            elo = (belo + welo) / 2
            opening = headers.get('Opening')
        except:
            eprint('skip')
            continue

        if elo < 1500:
            continue

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

        count += 1
pgn.close()
engine.close()
