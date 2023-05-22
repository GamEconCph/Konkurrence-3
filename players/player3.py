import numpy as np
from game_tournament.game import Player # overarching player class

class player(Player):
    name = 'Undercutting Bastard'
    
    def play(self, f_profit_own, f_profit_opponent, pmin, pmax, history_own, history_opponent, discount_factor):
        T = len(history_own)

        if T == 0: 
            # initial play 
            p = pmax * 0.999 
        else: 
            pj_lag = history_opponent[-1]
            
            # first idea: undercut by 10% 
            p = pj_lag * 0.9 
            
            # check if the price has gone too low 
            if (p - pmin) / (pmax - pmin) < 0.1: 
                # hike to maximum (hoping the friend follows)
                p = pmax
        
        p = np.clip(p, pmin, pmax)
        
        return p