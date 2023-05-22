import numpy as np
from game_tournament.game import Player 

class player(Player):
    name = 'Randawg'
    
    def play(self, f_profit_own, f_profit_opponent, pmin, pmax, history_own, history_opponent, discount_factor):
        
        p = np.random.uniform(pmin, pmax)
        
        return p 