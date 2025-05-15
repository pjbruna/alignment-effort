import numpy as np
from math import comb
import matplotlib.pyplot as plt
from models import *

n = 4
m = n**2
r = 0.5 # SDIC?
v = 2 / comb(m, 2)
t = 2 * (m**2)
lambda_values = np.linspace(0, 1, 21)


### Run model ###

count = 0
lex = []
mi = []
for l in lambda_values:
  print(f'Run: #{count}')

  mod = JointSpeakerAlignment(signal=n, referent=m, density=r)
  result = mod.run_to_equilibrium(prob=v, lam=l, stop=t)
  print(mod.energy_function(result[0], l))

  # Effective lexicon size:
  lexicon = result[0].sum(axis=0)
  lexicon[lexicon > 0] = 1
  lex.append(np.sum(lexicon)/n**2)

  # Mutual information: I(S1, S2, R) = Hnorm(S1, S2) - Hnrom(S1, S2 | R)
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

plt.show()

# plt.savefig("figures/ext_lexicon_and_MI.png")
plt.clf()
