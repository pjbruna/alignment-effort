import numpy as np
from math import comb
import matplotlib.pyplot as plt
from models import *

n = 4
m = n
r = 0.5 # SDIC?
v = 2 / comb(m**2, 2)
t = 2 * ((m**2)**2)
lambda_values = np.linspace(0, 1, 21)

### Run model ###
count = 0
lex = []
mi = []
for l in lambda_values:
  print(f'Run: #{count}')

  mod = ReferentialAlignment(signal=n, referent=m, density=r)
  result = mod.run_to_equilibrium(prob=v, lam=l, stop=t)

  # Effective lexicon size:
  lexicon = result[0].sum(axis=0).sum(axis=0)
  lexicon[lexicon > 0] = 1
  lex.append(np.sum(lexicon)/n**2)

  # Mutual information: I(S1, S2, R) = Hnorm(S1, S2) - Hnrom(S1, S2 | R)
  s1 = result[0].mean(axis=1)
  s2 = result[0].mean(axis=0)

  s1_mi = mutual_information(s1)
  s2_mi = mutual_information(s2)

  mi.append(np.mean([s1_mi, s2_mi]))

  count += 1


### Plot effective lexicon size and (normalized) mutual information ###

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].plot(lambda_values, lex, '.-')
axs[0].set_xlabel("Lambda")
axs[0].set_ylabel("Effective Lexicon Size")

axs[1].plot(lambda_values, mi, '.-')
axs[1].set_xlabel("Lambda")
axs[1].set_ylabel("MI")

plt.savefig("figures/4d_lexicon_and_MI.png")
plt.clf()
