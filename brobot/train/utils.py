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

def get_mobility(board, pieces, num):
    x = []
    pieces = list(pieces)
    for i in range(num):
        if len(pieces) - 1 < i:
            x.append([0, 0, 0, 0])
            continue
        mir = 8
        mar = 0
        mif = 8
        maf = 0
        index = pieces[i]
        ora = chess.square_rank(index)
        ofa = chess.square_file(index)
        for attack in board.attacks(index):
            sr = chess.square_rank(attack)
            sf = chess.square_file(attack)
            if sr < mir:
                mir = sr
            if sr > mar:
                mar = sr
            if sf < mif:
                mif = sf
            if sf > maf:
                maf = sf
        
        x.append([
                (mir - ora) / 7., 
                (mar - ora) / 7., 
                (mif - ofa) / 7., 
                (maf - ofa) / 7.
        ])
    return x

def get_move_rep(board, move):
    from_square = move.from_square
    to_square = move.to_square
    is_capture = board.is_capture(move)
    capture_piece = None
    if board.is_en_passant(move):
        capture_piece = 0
    else:
        capture_piece = board.piece_at(to_square).piece_type - 1 if is_capture else 10
    capture_piece_type = tf.keras.backend.one_hot(capture_piece, 6).numpy()
    piece_type = tf.keras.backend.one_hot(board.piece_at(from_square).piece_type - 1, 6).numpy()
    from_pos = [
        (chess.square_file(from_square) - 3.5) / 3.5,
        (chess.square_rank(from_square) - 3.5) / 3.5,
    ]
    to_pos = [
        (chess.square_file(to_square) - 3.5) / 3.5,
        (chess.square_rank(to_square) - 3.5) / 3.5,
    ]
    rank = 0

    num_attackers = len(board.attackers(not board.turn, from_square))
    num_defenders = len(board.attackers(not board.turn, to_square))
    num_attackers2 = len(board.attackers(board.turn, from_square))
    num_defenders2 = len(board.attackers(board.turn, to_square))
    rep = [
        float(is_capture),
        *capture_piece_type,
        num_attackers,
        num_defenders,
        num_attackers2,
        num_defenders2,
        *piece_type,
        *from_pos,
        *to_pos,
        *tf.keras.backend.one_hot(move.promotion - 1 if move.promotion else 10, 6).numpy(), # x > num_c gives empty vector
        #rank,
    ]
    return np.array(rep).astype('float32')

def get_pos_rep(board):
    castling = [
        int(board.has_kingside_castling_rights(chess.WHITE)),
        int(board.has_queenside_castling_rights(chess.WHITE)),
        int(board.has_kingside_castling_rights(chess.BLACK)),
        int(board.has_queenside_castling_rights(chess.BLACK)),
    ]

    wq = board.pieces(chess.QUEEN, chess.WHITE)
    wr = board.pieces(chess.ROOK, chess.WHITE)
    wb = board.pieces(chess.BISHOP, chess.WHITE)
    wn = board.pieces(chess.KNIGHT, chess.WHITE)
    wp = board.pieces(chess.PAWN, chess.WHITE)
    bq = board.pieces(chess.QUEEN, chess.BLACK)
    br = board.pieces(chess.ROOK, chess.BLACK)
    bb = board.pieces(chess.BISHOP, chess.BLACK)
    bn = board.pieces(chess.KNIGHT, chess.BLACK)
    bp = board.pieces(chess.PAWN, chess.BLACK)

    material = [
        len(wq),
        len(wr),
        len(wb),
        len(wn),
        len(wp),
        len(bq),
        len(br),
        len(bb),
        len(bn),
        len(bp),
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
        *get_pieces(board, wq, 1),
        *get_pieces(board, bq, 1),
        *get_pieces(board, wn, 2),
        *get_pieces(board, bn, 2),
        *get_pieces(board, wb, 2),
        *get_pieces(board, bb, 2),
        *get_pieces(board, wr, 2),
        *get_pieces(board, br, 2),
        *get_pieces(board, wp, 8),
        *get_pieces(board, bp, 8),
    ]

    mobility = [
        *get_mobility(board, wq, 1),
        *get_mobility(board, bq, 1),
        *get_mobility(board, wr, 2),
        *get_mobility(board, br, 2),
        *get_mobility(board, wb, 2),
        *get_mobility(board, bb, 2)
    ]

    attack_map = get_attack_map(board, not board.turn)
    defend_map = get_attack_map(board, board.turn)

    general_features = np.array([board.turn, board.is_check(), *castling, *material, *material_diff]).flatten().astype('float32')
    piece_features = np.array(pieces).flatten().astype('float32')
    mobility_features = np.array(mobility).flatten().astype('float32')
    square_features = np.array([*attack_map, *defend_map]).flatten().astype('float32')

    return [general_features, piece_features, mobility_features, square_features]
