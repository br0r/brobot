import sys
from enum import Enum
from collections import namedtuple
import time
import numpy as np
from chess.polyglot import zobrist_hash
from brobot.train.utils import get_pos_rep
from brobot.engine.evaluators.net_evaluator import get_moves_pred

class EntryFlag(Enum):
    EXACT = 1
    LOWERBOUND = 2
    UPPERBOUND = 3

Entry = namedtuple('Entry', 'value depth flag move')

def get_move_sort_score(board, move, color, best=None):
    score = 0
    if board.is_capture(move):
        score += 5000
    if board.gives_check(move):
        score += 1000
    if board.is_attacked_by(not color, move.from_square):
        score += 2000
    return score

def negamax(engine, depth, alpha, beta, color, root=False, prob=1.0, curr_depth=0):
    prob_threshold = engine.prob_threshold
    alphaorig = alpha
    board = engine.board
    evaluator = engine.evaluator
    transition_table = engine.transition_table
    moves = list(board.legal_moves)
    skip_cache = False

    if False and root:
        tmp = []
        for move in moves:
            board.push(move)
            if board.is_checkmate() or not board.is_game_over():
                tmp.append(move)
            board.pop()

        if len(tmp) < len(moves) and len(tmp) > 0:
            moves = tmp
            skip_cache = True
    
    h = zobrist_hash(board)
    # ttEntry lookup
    ttEntry = transition_table.get(h)
    if not skip_cache and ttEntry and ttEntry.depth >= depth:
        if ttEntry.flag == EntryFlag.EXACT:
            return (ttEntry.value, ttEntry.move, ttEntry.depth)
        elif ttEntry.flag == EntryFlag.LOWERBOUND:
            alpha = max(alpha, ttEntry.value)
        elif ttEntry.flag == EntryFlag.UPPERBOUND:
            beta = min(beta, ttEntry.value)

        if alpha >= beta:
            return (ttEntry.value, ttEntry.move, ttEntry.depth)

    if prob < prob_threshold:
        return [quiesce(board, evaluator, alpha, beta, color), None, curr_depth]

    _max = [-99999, None, curr_depth]
    if True and prob > (prob_threshold):
        # Pred sort
        moves = get_moves_pred(board, moves, h=h)
        moves.sort(key=lambda x: x[0], reverse=True)
    else:
        moves = sorted(moves, key=lambda x: get_move_sort_score(board, x, color), reverse=True)
        moves = [(1.0 / len(moves), x) for x in moves]

    for move in moves:
        (cprob, move) = move
        board.push(move)
        (score, _, node_depth) = negamax(engine, depth - 1, -beta, -alpha, -color, prob=prob*cprob, curr_depth=curr_depth+1)
        score = -score
        board.pop()
        if score > _max[0]:
            _max = [score, move, node_depth]
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

def quiesce(board, evaluator, alpha, beta, color, depth=10):
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
