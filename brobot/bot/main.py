import os
import berserk
from brobot.engine import Engine, evaluators
from brobot.bot.game import Game

token = os.getenv('API_TOKEN'); 
bot_id = os.getenv('BOT_ID');
games = {}

def start():
    session = berserk.TokenSession(token)
    client = berserk.Client(session)
    is_white = None
    game_id = None

    def should_accept(challenge):
        standard = challenge['variant']['key'] == 'standard'
        rated = challenge['rated']
        return standard and not rated 

    for event in client.bots.stream_incoming_events():
        print(event)
        if event['type'] == 'challenge':
            challenge = event['challenge']
            if should_accept(challenge):
                client.bots.accept_challenge(challenge['id'])
            else:
                client.bots.decline_challenge(challenge['id'])
        elif event['type'] == 'gameStart':
            gameEvent = event['game']
            #engine = Engine(evaluators.simple_evaluator)
            engine = Engine(evaluators.net_evaluator, depth=1)
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
            #client.challenges.create('jrti_bot', False, clock_limit=60*5, clock_increment=0)
