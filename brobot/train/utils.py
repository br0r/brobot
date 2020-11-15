from __future__ import print_function
import sys
import numpy as np
import chess

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def get_pieces(pieces, num):
    x = []
    pieces = list(pieces)
    for i in range(num):
        if len(pieces) - 1 < i:
            x.append([0, 0, 0])
            continue
        index = pieces[i]
        nfile = (chess.square_file(index) - 3.5) / 3.5
        nrank = (chess.square_rank(index) - 3.5) / 3.5
        x.append([1, nfile, nrank])
    return x

pieces = ['p','n','b','r','q','k','P','N','B','R','Q','K']
piecesm = {}
for i in range(len(pieces)):
    piecesm[pieces[i]] = i

def get_train_row_old(board):
    piece_map = board.piece_map()
    rep = np.zeros((8 * 8, 12))

    for key in piece_map:
        val = str(piece_map[key])
        ind = piecesm[val]
        label = None
        if ind <= 5 and not board.turn: #black piece, black to move
            label = 1
        if ind <= 5 and board.turn: #black piece, white to move
            label = -1
        if ind > 5 and board.turn: #white piece, white to move
            label = 1
        if ind > 5 and not board.turn: #white piece, black to move
            label = -1

        rep[key][ind] = label

    rep = rep.flatten()
    return rep

def get_train_row(board):
    castling = [
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    ]

    material = [
        len(board.pieces(chess.QUEEN, chess.WHITE)),
        len(board.pieces(chess.ROOK, chess.WHITE)) / 2,
        len(board.pieces(chess.BISHOP, chess.WHITE)) / 2,
        len(board.pieces(chess.KNIGHT, chess.WHITE)) / 2,
        len(board.pieces(chess.PAWN, chess.WHITE)) / 8,
        len(board.pieces(chess.QUEEN, chess.BLACK)),
        len(board.pieces(chess.ROOK, chess.BLACK)) / 2,
        len(board.pieces(chess.BISHOP, chess.BLACK)) / 2,
        len(board.pieces(chess.KNIGHT, chess.BLACK)) / 2,
        len(board.pieces(chess.PAWN, chess.BLACK)) / 8
    ]

    pieces = [
        *get_pieces(board.pieces(chess.KING, chess.WHITE), 1),
        *get_pieces(board.pieces(chess.QUEEN, chess.WHITE), 1),
        *get_pieces(board.pieces(chess.ROOK, chess.WHITE), 2),
        *get_pieces(board.pieces(chess.BISHOP, chess.WHITE), 2),
        *get_pieces(board.pieces(chess.KNIGHT, chess.WHITE), 2),
        *get_pieces(board.pieces(chess.PAWN, chess.WHITE), 8),
        *get_pieces(board.pieces(chess.KING, chess.BLACK), 1),
        *get_pieces(board.pieces(chess.QUEEN, chess.BLACK), 1),
        *get_pieces(board.pieces(chess.ROOK, chess.BLACK), 2),
        *get_pieces(board.pieces(chess.BISHOP, chess.BLACK), 2),
        *get_pieces(board.pieces(chess.KNIGHT, chess.BLACK), 2),
        *get_pieces(board.pieces(chess.PAWN, chess.BLACK), 8),
    ]

    general_features = [board.turn, *castling, *material]
    piece_features = np.array(pieces).flatten().astype('float32')

    return [general_features, piece_features]
