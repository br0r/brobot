import time
import chess
from brobot.engine import Engine, evaluators

def main():
    fen = None
    engine = Engine(evaluators.net_evaluator, name="net", depth=1, fen=fen)
    engine2 = Engine(evaluators.simple_evaluator, name="simple", depth=2, fen=fen)
    #engine = Engine(evaluators.simple_evaluator, name="simple", depth=3, fen=fen)
    #engine2 = Engine(evaluators.stupid_evaluator, name="stupid", depth=1, fen=fen)

    engine.color= chess.WHITE
    engine2.color = chess.BLACK
    turn = 1
    game = chess.pgn.Game()
    game.headers["Event"] = "Testgame"
    game.headers["White"] = engine.name
    game.headers["Black"] = engine2.name

    movehistory = []
    while engine.board.is_game_over() == False:
        t = time.time()
        if turn == 1:
            score, move = engine.find_best_move()
            engine.make_move(move)
            engine2.make_move(move)
        else:
            score, move = engine2.find_best_move()
            engine.make_move(move)
            engine2.make_move(move)

        dt = time.time() - t

        movehistory.append(move)
        print(move, dt)

        turn = turn * -1

    print("Done, result", engine.board.result())
    game.add_line(movehistory)
    print(game)


if __name__ == '__main__':
    main()
