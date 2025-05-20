import numpy as np
from game_tournament.game import Player # overarching player class

from scipy.optimize import minimize
from scipy.optimize import fsolve

class player(Player):
    name = 'Soft Trigger'

    # this player implements a trigger-type strategy which has some measure of 
    # forgiveness built in. 
    
    def play(self, f_profit_own, f_profit_opponent, pmin, pmax, history_own, history_opponent, discount_factor):

        T = len(history_own)

        p0 = (pmax - pmin) / 2. # initial guess for the optimization routine

        # find the optimal static cartel price 
        def f_pi_cartel(p_vec): 
            p = p_vec[0]
            pi_cartel = f_profit_own(p,p) + f_profit_opponent(p,p)
            return -pi_cartel # we *minimize* the negative profit 

        p_cartel = minimize_scalar_f(f_pi_cartel, p0, pmin, pmax)

        if T == 0: 
            # initial play: just the cartel equilibrium 
            p = p_cartel 
        else: 
            plag_opponent = history_opponent[-1]
            
            if np.abs(p_cartel-plag_opponent) < 0.1: # allow some slack 
                # cooperation phase 
                # if the cartel price is close to the last price, keep cooperating 
                p = p_cartel 
            else: 
                # punishment phase 
                p1_nash,p2_nash = compute_nash_simple(f_profit_own, f_profit_opponent, pmin, pmax)
                p = p1_nash
                
        
        # safety precautions 
        if not np.isscalar(p):
            p = (pmax - pmin) / 2. # if something went wrong, just play the middle price

        p = np.clip(p, pmin, pmax) # cut to fit inside the permitted interval

        return p
    

# Helper functions for the player class

def compute_nash_simple(f_profit_own, f_profit_opponent, pmin, pmax):
    '''
    INPUTS: 
        f_profit_own: Profit of player 1, inputs: (p1,p2)
        f_profit_opponent: Profit of player 2, inputs: (p2,p1)
        pmin: (scalar) minimum price
        pmax: (scalar) maximum price
    OUTPUTS: (p1, p2)
        p1: (scalar) Nash equilibrium price of player 1
        p2: (scalar) Nash equilibrium price of player 2
    '''
    p0 = (pmax - pmin) / 2. # initial guess for the optimization routine
    def BR2(p1): 
        f = lambda p2 : -f_profit_opponent(p2[0],p1) # opponent's profit takes p2 as first input! 
        p2 = minimize_scalar_f(f, p0, pmin, pmax)
        return p2 
    
    def BR1(p2): 
        f = lambda p1 : -f_profit_own(p1[0],p2)
        p1 = minimize_scalar_f(f, p0, pmin, pmax)
        return p1
    
    # vectorized version of the best response function
    def BR_fxp(p): 
        p1 = BR1(p[0])
        p2 = BR2(p1)
        return np.array([p1,p2]) - p # we're looking for a zero of this function
    
    # use fsolve to find the Nash equilibrium
    pstart = np.array([p0,p0]) # starting guess for fsolve
    p1, p2 = fsolve(BR_fxp, pstart) # find the Nash equilibrium
    p1 = np.clip(p1, pmin, pmax) # cut to fit inside the permitted interval
    p2 = np.clip(p2, pmin, pmax) # cut to fit inside the permitted interval

    return p1, p2


def minimize_scalar_f(f:callable, x0:float, pmin:float, pmax:float) -> float: 
    x0 = np.array([x0]) # convert to vector (of length 1) for minimize
    res = minimize(f, x0=x0, bounds=[(pmin,pmax)], 
                   tol=1e-6, options={'maxiter': 20})
    p_scalar = res.x[0] # minimize returns a vector
    return p_scalar 