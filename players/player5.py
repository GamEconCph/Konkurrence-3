import numpy as np
from game_tournament.game import Player # overarching player class
from scipy.optimize import minimize

class player(Player):
    name = 'Trigger'

    # this player implements a trigger-type strategy which has some measure of 
    # forgiveness built in. 
    
    def play(self, f_profit_own, f_profit_opponent, pmin, pmax, history_own, history_opponent, discount_factor):

        T = len(history_own)

        p0 = (pmax - pmin) / 2. # initial guess for the optimization routine

        # find the optimal static cartel price 
        def f_pi_cartel(p_vec): 
            p = p_vec[0]
            return -f_profit_own(p,p) - f_profit_opponent(p,p)

        p_cartel = minimize_scalar_f(f_pi_cartel, p0, pmin, pmax)

        if T == 0: 
            # initial play: just the cartel equilibrium 
            p = p_cartel 
        else: 
            plag_2 = history_opponent[-1]
            
            if np.abs(p_cartel - plag_2) < 0.1: 
                # cooperation phase 
                # if the cartel price is close to the last price, keep cooperating 
                p = p_cartel 
            else: 
                # punishment phase 
                p1_nash,p2_nash = compute_nash_simple(f_profit_own, f_profit_opponent, pmin, pmax)
                p = p1_nash
                
        assert np.isscalar(p), f'Something went wrong. p is not a scalar: {p}'
        p = np.clip(p, pmin, pmax) # cut to fit inside the permitted interval
        return p
    
def compute_nash_simple(f_profit_own, f_profit_opponent, pmin, pmax):
    p0 = (pmax - pmin) / 2. # initial guess for the optimization routine
    def BR2(p1): 
        f = lambda p2 : -f_profit_opponent(p2[0],p1) # opponent's profit takes p2 as first input! 
        p2 = minimize_scalar_f(f, p0, pmin, pmax)
        return p2 
    
    def BR1(p2): 
        f = lambda p1 : -f_profit_own(p1[0],p2)
        p1 = minimize_scalar_f(f, p0, pmin, pmax)
        return p1
    
    # Iterative Best Response (IBR) Algorithm 
    p2 = p0 # starting guess for IBR 
    for i in range(10): 
        p1 = BR1(p2)
        p2 = BR2(p1)

    return p1, p2 


def minimize_scalar_f(f, x0, pmin, pmax): 
    x0 = np.array([x0]) # minimize wants a vector input
    res = minimize(f, x0=x0, bounds=[(pmin,pmax)], 
                   tol=1e-6, options={'maxiter': 100})
    p_scalar = res.x[0] # minimize returns a vector
    return p_scalar 