# Konkurrence 3: Gentaget Priskonkurrence

I denne konkurrence skal du skrive en spillerfunktion, der kan spille et gentaget priskonkurrencespil. 
Når du er færdig, skal du navngive din fil player.py, og lægge den i mappen ./submission/ og committe. **Husk** at sætte dens property name = 'xxx', hvor xxx skal være et navn, som du vil bruge i samtlige konkurrencer i kurset.

Den medfølgende notebook, `short.ipynb` loader spillerfunktioner fra mappen ./players/ og opstiller en turnering, hvor de kan spille imod hinanden. I `long.ipynb` finder du mere inspiration til, hvordan du kan opstille din funktion. 

# Spillerfunktionen

Spillerfunktionen skal tage flg. inputs

* `f_profit_own`: funktion af to inputs, som giver din profit, `f_profit_own(p1, p2)`,
* `f_profit_opponent`: samme, men for modstanderen, dog med *modsat rækkefølge* af inputs: `f_profit_opponent(p2, p1)`,
* `pmin`: float, mindste tilladte pris,
* `pmax`: float, største tilladte pris,
* `discount_factor`: float, diskonteringsfaktorn (mellem 0.0 og 1.0)

Nedenfor ser du et eksempel på en spillerfunktion, som vælger en tilfældig handling.

```Python
name = 'Randawg' # remember to use the same name in all competitions... 
def play(self, f_profit_own, f_profit_opponent, pmin, pmax, history_own, history_opponent, discount_factor):
  p = np.random.uniform(pmin, pmax) # rather ineffecient strategy
  return p 
```

**Tilfældighed:** Du må gerne bruge tilfældighed, fx `np.random.uniform()` (eller `.normal()`) til at vælge blandt flere kandidater. 
Turneringen vil blive gentaget 100 gange mellem dig og din modstander for at midle sådan tilfældighed ud.

## Historikken 

Du kan se i eksemplet `player5.py`, som foretager sig mere avancerede ting med historikken. 

```Python
name = 'Tit for Tat'
def play(self, f_profit_own, f_profit_opponent, pmin, pmax, history_own, history_opponent, discount_factor):
  T = len(history_own) # antal periode, der er spillet 
  if T == 0: 
    # første runde i spillet
    p2 = (pmax-pmin)/2.0 # some (dumb) guess of opponent's strategy
    f = lambda p1: -f_profit_own(p1,p2)
    res = minimize_scalar(f, p2, bounds=(pmin,pmax), options={'maxiter': 20})
    p = res.x
  else: 
    pj_lag = history_opponent[-1] # the last thing our opponent played 
    p = pj_lag # we play the same 
  p = np.clip(p, pmin, pmax) # ensure we have not set something illegal
  return p 
```

Denne funktion har en særlig undtagelse for den første periode i spillet, hvor der naturligvis ikke er nogen historik at betinge på. 

Hvis du bruger numeriske solvere, så husk venligst at sætte `maxiter`, så den ikke bruger håbløst meget tid :) 


