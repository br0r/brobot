import time
import os
import berserk
from brobot.engine import Engine, evaluators
from brobot.bot.game import Game

token = os.getenv('API_TOKEN');
bot_id = os.getenv('BOT_ID')
TRIGGER_REMATCH = bool(int(os.getenv('REMATCH', 1)))
games = {}
challengers = {}

def start():
    session = berserk.TokenSession(token)
    client = berserk.Client(session)
    is_white = None
    game_id = None

    def should_accept(challenge):
        return True
        standard = challenge['variant']['key'] == 'standard'
        rated = challenge['rated']
        return standard and not rated 

    for event in client.bots.stream_incoming_events():
        print(event)
        if event['type'] == 'challenge':
            challenge = event['challenge']
            # We did challenge
            if challenge['challenger']['id'] == bot_id:
                continue
            if should_accept(challenge):
                client.bots.accept_challenge(challenge['id'])
            else:
                client.bots.decline_challenge(challenge['id'])
        elif event['type'] == 'gameStart':
            gameEvent = event['game']
            #engine = Engine(evaluators.simple_evaluator)
            engine = Engine(evaluators.net_evaluator, depth=3)
            game = Game(client, gameEvent['id'], engine, bot_id)
            games[gameEvent['id']] = game
            game.start()
        elif event['type'] == 'gameFinish':
            gameEvent = event['game']
            game_id = gameEvent['id']
            client.bots.post_message(game_id, 'GG')
            if game_id in games:
                game = games[game_id]
                game.join()
                del games[game_id]

            if TRIGGER_REMATCH:
                game = client.games.export(game_id)
                players = game['players']
                print(players)
                white = players['white']['user'] if 'user' in players['white'] else None
                black = players['black']['user'] if 'user' in players['black'] else None
                if not white or not black:
                    # ai
                    ailevel = players['white']['aiLevel'] if white is None else players['black']['aiLevel']
                    time.sleep(1)
                    client.challenges.create_ai(ailevel, clock_limit=60*30, clock_increment=30)
                else:
                    opponent = white if white['id'] != bot_id else black
                    time.sleep(1)
                    client.challenges.create(opponent['id'], False, clock_limit=60*5, clock_increment=5)
