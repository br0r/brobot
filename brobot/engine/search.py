from enum import Enum
from collections import namedtuple
import time
import numpy as np
from chess.polyglot import zobrist_hash

class EntryFlag(Enum):
    EXACT = 1
    LOWERBOUND = 2
    UPPERBOUND = 3

Entry = namedtuple('Entry', 'value depth flag move')

def get_move_sort_score(board, move, color):
    score = 0
    if board.is_capture(move):
        score += 2000
    if board.gives_check(move):
        score += 3000
    if board.is_attacked_by(not color, move.from_square):
        score += 1000
    if board.is_castling(move):
        score += 500
    return score

def negamax(engine, depth, alpha, beta, color):
    alphaorig = alpha
    curr_depth = engine.depth - depth
    board = engine.board
    evaluator = engine.evaluator
    transition_table = engine.transition_table
    
    h = zobrist_hash(board)
    # ttEntry lookup
    ttEntry = transition_table.get(h)
    if ttEntry and ttEntry.depth >= depth:
        if ttEntry.flag == EntryFlag.EXACT:
            return (ttEntry.value, ttEntry.move)
        elif ttEntry.flag == EntryFlag.LOWERBOUND:
            alpha = max(alpha, ttEntry.value)
        elif ttEntry.flag == EntryFlag.UPPERBOUND:
            beta = min(beta, ttEntry.value)

        if alpha >= beta:
            return (ttEntry.value, ttEntry.move)

    if depth == 0:
        return [quiesce(board, evaluator, alpha, beta, color), None]

    _max = [-99999, []]
    moves = board.legal_moves
    moves = sorted(moves, key=lambda x: get_move_sort_score(board, x, color), reverse=True)
    for move in moves:
        board.push(move)
        score = -negamax(engine, depth - 1, -beta, -alpha, -color)[0]
        board.pop()
        if score > _max[0]:
            _max = [score, move]
        alpha = max(alpha, _max[0])
        if alpha >= beta:
            break

    # Store ttEntry
    value = _max[0]
    if value <= alphaorig:
        flag = EntryFlag.UPPERBOUND
    elif value >= beta:
        flag = EntryFlag.LOWERBOUND
    else:
        flag = EntryFlag.EXACT
    ttEntry = Entry(value=value, depth=depth, flag=flag, move=_max[1])
    transition_table[h] = ttEntry
    return _max

def quiesce(board, evaluator, alpha, beta, color, depth=100):
    standpat = color * evaluator(board)
    if depth == 0:
        return standpat
    if standpat >= beta:
        return beta
    if alpha < standpat:
        alpha = standpat
    
    for child in board.legal_moves:
        if not board.is_capture(child):
            continue
        board.push(child)
        score = -quiesce(board, evaluator, -beta, -alpha, -color, depth - 1)
        board.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha
