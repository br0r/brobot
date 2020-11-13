import numpy as np

pieces = ['p','n','b','r','q','k','P','N','B','R','Q','K']
piecesm = {}
for i in range(len(pieces)):
    piecesm[pieces[i]] = i

def get_train_row(board):
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
        if ind > 5 and not board.turn: #black piece, black to move
            label = -1

        rep[key][ind] = label

    rep = rep.flatten()
    return rep
