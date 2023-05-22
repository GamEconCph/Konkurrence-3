import numpy as np
from game_tournament.game import Player # overarching player class

class player(Player):
    name = 'pmin'
    
    def play(self, f_profit_own, f_profit_opponent, pmin, pmax, history_own, history_opponent, discount_factor):
        return pmin