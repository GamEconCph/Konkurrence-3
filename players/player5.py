import numpy as np
from game_tournament.game import Player # overarching player class
from scipy.optimize import minimize_scalar

class player(Player):
    name = 'Trigger'

    # this player implements a trigger-type strategy which has some measure of 
    # forgiveness built in. 
    
    def play(self, f_profit_own, f_profit_opponent, pmin, pmax, history_own, history_opponent, discount_factor):

        T = len(history_own)

        # find the optimal static cartel price 
        f_pi_cartel = lambda p : -f_profit_own(p,p) - f_profit_opponent(p,p)
        res = minimize_scalar(f_pi_cartel, bounds=(pmin,pmax), options={'maxiter':20})
        p_cartel = res.x
        if T == 0: 
            # initial play: just the cartel equilibrium 
            p = p_cartel 
        else: 
            plag_2 = history_opponent[-1]
            if np.abs(p_cartel - plag_2) < 0.1: 
                # if the cartel price is close to the last price, keep cooperating 
                p = p_cartel 
            else: 
                # otherwise, play some punishment 
                def BR2(p1): 
                    f = lambda p2 : -f_profit_opponent(p2,p1)
                    res = minimize_scalar(f, bounds=(pmin,pmax), options={'maxiter':20})
                    p2 = res.x
                    return p2 
                
                def BR1(p2): 
                    f = lambda p1 : -f_profit_own(p1,p2)
                    res = minimize_scalar(f, bounds=(pmin,pmax), options={'maxiter':20})
                    p1 = res.x
                    return p1
                
                # take some iterations on the best response mapping 
                p2 = plag_2
                for i in range(10): 
                    p1 = BR1(p2)
                    p2 = BR2(p1)

                # our final choice                 
                p = BR1(p2)

        assert np.isscalar(p), f'Something went wrong. p is not a scalar: {p}'
        p = np.clip(p, pmin, pmax) # cut to fit inside the permitted interval
        return p