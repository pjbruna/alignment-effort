import numpy as np
from math import comb
import matplotlib.pyplot as plt
from models import *

n = 36
r = 0.5 # SDIC?
v = 2 / comb(n, 2)
t = 2 * (n**2)
lambda_values = np.linspace(0, 1, 21)


### Run model ###

count = 0
lex = []
mi = []
for l in lambda_values:
  print(f'Run: #{count}')

  m = SignalEvolution(size=n, density=r)
  result = m.run_to_equilibrium(prob=v, lam=l, stop=t)
  print(m.energy_function(result[0], l))

  # Effective lexicon size:
  lexicon = result[0].sum(axis=0)
  lexicon[lexicon > 0] = 1
  lex.append(np.sum(lexicon)/n)

  # Mutual information: I(S, R)
  mi.append(mutual_information(result[0]))

  count += 1


### Plot effective lexicon size and (normalized) mutual information ###

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].plot(lambda_values, lex, '.-')
axs[0].set_xlabel("Lambda")
axs[0].set_ylabel("Effective Lexicon Size")

axs[1].plot(lambda_values, mi, '.-')
axs[1].set_xlabel("Lambda")
axs[1].set_ylabel("MI")

plt.savefig("figures/rep_lexicon_and_MI.png")
plt.clf()