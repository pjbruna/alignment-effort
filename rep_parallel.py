import numpy as np
from math import comb
import matplotlib.pyplot as plt
import pandas as pd
from models import *
import multiprocessing as mp

n = 36
r = 0.5 # SDIC?
v = 2 / comb(n, 2)
t = 2 * (n**2)
lambda_values = np.linspace(0, 1, 21)

def run_model(index):
  print(f'Running model: {index}')

  lex = []
  mi = []
  lam = []

  for l in lambda_values:
    print(f'Lambda: {l}')

    m = SignalEvolution(size=n, density=r)
    result = m.run_to_equilibrium(prob=v, lam=l, stop=t)

    # Effective lexicon size:
    lexicon = result[0].sum(axis=0)
    lexicon[lexicon > 0] = 1
    lex.append(np.sum(lexicon)/n)

    # Mutual information: I(S, R)
    mi.append(mutual_information(result[0]))

    # Store lambda and run
    lam.append(l)

  return lex, mi, lam

### Run model ###

if __name__ == "__main__":
    runs = 2

    # Create a multiprocessing pool
    with mp.Pool() as pool:        
        results = pool.map(run_model, range(runs))
    
    # Compile results
    lex = []
    mi = []
    lam = []
    run_num = []

    for i in range(runs):
        lex.append(results[i][0][0])
        mi.append(results[i][1][0])
        lam.append(results[i][2][0])
        run_num.extend(np.repeat(i+1, len(results[i][2][0])))
    
    ## STORE DATA ##

    log_lex = [item for sublist in lex for item in sublist]
    log_mi = [item for sublist in mi for item in sublist]
    log_lam = [item for sublist in lam for item in sublist]

    data = {
        'lexicon': log_lex,
        'mi': log_mi,
        'lambda': log_lam,
        'run': run_num
    }

    df = pd.DataFrame(data)
    df.to_csv(f'data/replication.csv', index=False)