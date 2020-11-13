import threading
import chess
import datetime
import time

class Game(threading.Thread):
    def __init__(self, client, game_id, engine, bot_id, **kwargs):
        super().__init__(**kwargs)
        self.bot_id = bot_id
        self.game_id = game_id
        self.client = client
        self.engine = engine
        self.stream = client.bots.stream_game_state(game_id)
        self.current_state = next(self.stream)

    def run(self):
        if self.current_state:
            is_white = self.current_state['white']['id'] == self.bot_id
            self.engine.color = chess.WHITE if is_white else chess.BLACK
            state = self.current_state['state']
            self.handle_state_change(state)
                
        for event in self.stream:
            if 'winner' in event:
                break
            if event['type'] == 'gameState':
                self.handle_state_change(event)
            elif event['type'] == 'chatLine':
                self.handle_chat_line(event)

    def handle_state_change(self, state):
        timeleft = state['wtime'] if self.engine.color == chess.WHITE else state['btime']
        if isinstance(timeleft, datetime.datetime):
            self.engine.set_timeleft((timeleft.hour * 3600 + timeleft.minute * 60 + timeleft.second))
        else:
            self.engine.set_timeleft(float(timeleft) / 1000)

        self.make_moves(state['moves'])
        if self.engine.board.turn == self.engine.color:
            self.make_move()

    def handle_chat_line(self, chat_line):
        print('chat', chat_line)
        pass

    def make_moves(self, moves):
        self.engine.board.reset()
        if not moves:
            return
        for move in moves.split(' '):
            uci_move = chess.Move.from_uci(move)
            self.engine.make_move(uci_move)

    def make_move(self):
        score, move = self.engine.find_best_move()
        print('MAKE_MOVE', move)
        time.sleep(0.2)
        self.client.bots.make_move(self.game_id, move)

