import numpy as np
from game_tournament.game import Player # overarching player class

# custom libraries 
from scipy.optimize import minimize

class player(Player):
    name = 'Best responder'
    
    def play(self, f_profit_own, f_profit_opponent, pmin, pmax, history_own, history_opponent, discount_factor):
        T = len(history_own)

        def BR(p_opponent): 
            f = lambda p : -f_profit_own(p, p_opponent)
            p = minimize_scalar_f(f, x0=p_opponent, pmin=pmin, pmax=pmax)
            return p 

        if T == 0: 
            # initial play 
            p_opponent = (pmax + pmin) / 2
        else: 
            p_opponent = history_opponent[-1]

        try: 
            p = BR(p_opponent)
        except: 
            # if something goes wrong, just play the middle price
            p = (pmax + pmin) / 2
        
        # for savety 
        p = np.clip(p, pmin, pmax)
        
        return p
    
# Helper functions for the player class
def minimize_scalar_f(f:callable, x0:float, pmin:float, pmax:float) -> float: 
    # Wrapper function for minimize, which assumes that the input is an *array* 
    x0 = np.array([x0]) # convert to vector (of length 1) for minimize
    res = minimize(f, x0=x0, bounds=[(pmin,pmax)], 
                   tol=1e-6, options={'maxiter': 100})
    p_scalar = res.x[0] # minimize returns a vector
    return p_scalar 