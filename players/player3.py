import numpy as np
from game_tournament.game import Player # overarching player class

class player(Player):
    name = 'Tit-for-tat'
    
    def play(self, f_profit_own, f_profit_opponent, pmin, pmax, history_own, history_opponent, discount_factor):
        # Starts at some price, then copies opponent's action
        T = len(history_own)

        if T == 0: 
            # initial play 
            p = (pmax+pmin)/2 # just some permitted starting price 
        else: 
            # dynamic play 
            p = history_opponent[-1]

        p = np.clip(p, pmin, pmax)
        
        return p 