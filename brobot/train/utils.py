from __future__ import print_function
import sys
import numpy as np
import tensorflow as tf
import chess

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

values = {
    chess.PAWN: 100,
    chess.KNIGHT: 416,
    chess.BISHOP: 441,
    chess.ROOK: 663,
    chess.QUEEN: 1292,
    chess.KING: 300
}

def get_min_and_max(board, arr):
    pieces_ = [board.piece_type_at(x) for x in arr]
    #pieces_ = [x for x in pieces_ if x != chess.KING]
    if not pieces_:
        return 0.0, 0.0
    mi = min([values[piece] for piece in pieces_]) / 100.0
    ma = max([values[piece] for piece in pieces_]) / 100.0
    return mi, ma

def get_pieces(board, pieces, num):
    x = []
    pieces = list(pieces)
    for i in range(num):
        if len(pieces) - 1 < i:
            x.append([0, 0, 0, 0, 0, 0, 0])
            continue
        index = pieces[i]
        nfile = (chess.square_file(index) - 3.5) / 3.5
        nrank = (chess.square_rank(index) - 3.5) / 3.5
        lowest_attacker, highest_attacker = get_min_and_max(board, board.attackers(not board.color_at(index), index))
        lowest_defender, highest_defender  = get_min_and_max(board, board.attackers(board.color_at(index), index))
        x.append([1, nfile, nrank, lowest_attacker, lowest_defender, highest_attacker, highest_defender])
    return x

def get_attack_map(board, color):
    map_ = np.zeros((64,2))
    for i in range(64):
        lowest_attacker, highest_attacker = get_min_and_max(board, board.attackers(color, i))
        map_[i] = [lowest_attacker, highest_attacker]
    return map_

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
        int(board.has_kingside_castling_rights(chess.WHITE)),
        int(board.has_queenside_castling_rights(chess.WHITE)),
        int(board.has_kingside_castling_rights(chess.BLACK)),
        int(board.has_queenside_castling_rights(chess.BLACK)),
    ]

    material = [
        len(board.pieces(chess.QUEEN, chess.WHITE)),
        len(board.pieces(chess.ROOK, chess.WHITE)),
        len(board.pieces(chess.BISHOP, chess.WHITE)),
        len(board.pieces(chess.KNIGHT, chess.WHITE)),
        len(board.pieces(chess.PAWN, chess.WHITE)),
        len(board.pieces(chess.QUEEN, chess.BLACK)),
        len(board.pieces(chess.ROOK, chess.BLACK)),
        len(board.pieces(chess.BISHOP, chess.BLACK)),
        len(board.pieces(chess.KNIGHT, chess.BLACK)),
        len(board.pieces(chess.PAWN, chess.BLACK))
    ]

    m = material
    material_diff = [
        m[0] - m[5],
        m[1] - m[6],
        m[2] - m[7],
        m[3] - m[8],
        m[4] - m[9],
    ]

    material = tf.keras.utils.normalize(material)[0]
    material_diff = tf.keras.utils.normalize(material_diff)[0]

    pieces = [
        *get_pieces(board, board.pieces(chess.KING, chess.WHITE), 1),
        *get_pieces(board, board.pieces(chess.KING, chess.BLACK), 1),
        *get_pieces(board, board.pieces(chess.QUEEN, chess.WHITE), 1),
        *get_pieces(board, board.pieces(chess.QUEEN, chess.BLACK), 1),
        *get_pieces(board, board.pieces(chess.KNIGHT, chess.WHITE), 2),
        *get_pieces(board, board.pieces(chess.KNIGHT, chess.BLACK), 2),
        *get_pieces(board, board.pieces(chess.BISHOP, chess.WHITE), 2),
        *get_pieces(board, board.pieces(chess.BISHOP, chess.BLACK), 2),
        *get_pieces(board, board.pieces(chess.ROOK, chess.WHITE), 2),
        *get_pieces(board, board.pieces(chess.ROOK, chess.BLACK), 2),
        *get_pieces(board, board.pieces(chess.PAWN, chess.WHITE), 8),
        *get_pieces(board, board.pieces(chess.PAWN, chess.BLACK), 8),
    ]

    attack_map = get_attack_map(board, not board.turn)
    defend_map = get_attack_map(board, board.turn)

    general_features = np.array([board.turn, board.is_check(), *castling, *material, *material_diff]).flatten().astype('float32')
    piece_features = np.array(pieces).flatten().astype('float32')
    square_features = np.array([*attack_map, *defend_map]).flatten().astype('float32')

    return [general_features, piece_features, square_features]
