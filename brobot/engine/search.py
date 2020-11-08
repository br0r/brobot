from collections import namedtuple
import time
import numpy as np
from chess.polyglot import zobrist_hash

Entry = namedtuple('Entry', 'move score depth')

def sort_moves(board, moves, transition_table):
    def score_move(move):
        if transition_table:
            board.push(move)
            hash_key = zobrist_hash(board)
            board.pop()
            entry = transition_table.get(hash_key)
            if entry:
                return entry.score

        score = int(board.is_capture(move))
        score += int(board.gives_check(move))
        return score

    sorted_moves = moves.copy()
    sorted_moves.sort(key=score_move, reverse=True)
    return sorted_moves

def iterative_deepening(board, depth, maximizingPlayer, evaluator, timelimit=None, transition_table=None):
    best = (-999999, None)
    legal_moves = list(board.legal_moves)
    np.random.shuffle(legal_moves)
    moves = sort_moves(board, legal_moves, transition_table)
    start = time.time()

    for d in range(1, depth if not timelimit else 100):
        dbest = (-999999, None)
        moves = sort_moves(board, legal_moves, transition_table)
        alpha = -99999
        beta = 99999
        for move in moves:
            board.push(move)

            hash_key = None
            if transition_table is not None:
                hash_key = zobrist_hash(board)
            
            score = None
            if hash_key and hash_key in transition_table:
                (_, _score, depth) = transition_table[hash_key]
                if depth >= d:
                    score = _score
                    print('Used transition table')

            if not score:
                score = alphabeta(board, d, maximizingPlayer, evaluator, transition_table, alpha, beta)

            if hash_key:
                entry = Entry(move=move, score=score, depth=d)
                transition_table[hash_key] = entry

            if score > dbest[0]:
                dbest = (score, move)

            board.pop()

            if maximizingPlayer:
                alpha = max(alpha, score)
                if alpha >= beta:
                    break
            else:
                beta = min(beta, score)
                if beta <= alpha:
                    break

            if timelimit:
                dt = time.time() - start
                left = timelimit - dt
                if left < 0:
                    print('DEPTH', d - 1, dt)
                    return best

        best = dbest
        if timelimit:
            dt = time.time() - start
            left = timelimit - dt
            if left < 1:
                print('DEPTH', d, dt)
                return best

    return best

def minimax(board, depth, maximizingPlayer, evaluator):
    if depth == 0:
        v = evaluator(board)
        if maximizingPlayer:
            return v
        else:
            return -v

    if maximizingPlayer:
        value = -99999
        for child in board.legal_moves:
            board.push(child)
            value = max(value, minimax(board, depth - 1, False, evaluator))
            board.pop()
        return value
    else:
        value = 99999
        for child in board.legal_moves:
            board.push(child)
            value = min(value, minimax(board, depth - 1, True, evaluator))
            board.pop()
        return value

def alphabeta(board, depth, maximizingPlayer, evaluator, transition_table=None, alpha=-99999, beta=99999):
    if depth == 0:
        v = evaluator(board)
        if maximizingPlayer:
            return v
        else:
            return -v

    hash_key = None
    if transition_table is not None:
        hash_key = zobrist_hash(board)
    
    score = None
    if hash_key and hash_key in transition_table:
        (_, _score, d) = transition_table[hash_key]
        if d >= depth:
            score = _score
            if maximizingPlayer:
                return score
            else:
                return -score

    if maximizingPlayer:
        value = -99999
        for child in board.legal_moves:
            board.push(child)
            value = max(value, alphabeta(board, depth - 1, False, evaluator, transition_table, alpha, beta))
            alpha = max(alpha, value)
            board.pop()
            if alpha >= beta:
                break
        return value
    else:
        value = 99999
        for child in board.legal_moves:
            board.push(child)
            value = min(value, alphabeta(board, depth - 1, True, evaluator, transition_table, alpha, beta))
            beta = min(beta, value)
            board.pop()
            if beta <= alpha:
                break
        return value
