import numpy as np
from math import comb
import matplotlib.pyplot as plt
from models import *

n = 15
r = 0.5 # SDIC?
v = 2 / comb(n, 2)
t = 3 * (n**2)
gamma_values = np.linspace(0, 1, 21)


### Run model ###

count = 0
l = []
mi = []
for g in gamma_values:
  print(f'Run: #{count}')

  m = JointSpeakerAlignment(size=n, density=r)
  result = m.run_to_equilibrium(prob=v, gamma=g, stop=t)
  print(m.energy_function(result, g))

  # Effective lexicon size:
  lexicon = result.sum(axis=0)
  lexicon[lexicon > 0] = 1
  l.append(np.sum(lexicon)/n**2)

  # s1_lexicon = result.sum(axis=1).sum(axis=0)
  # s1_lexicon[s1_lexicon > 0] = 1
  # s1_eff_lexicon = np.sum(s1_lexicon)/n

  # s2_lexicon = result.sum(axis=2).sum(axis=0)
  # s2_lexicon[s2_lexicon > 0] = 1
  # s2_eff_lexicon = np.sum(s2_lexicon)/n

  # l.append((s1_eff_lexicon + s2_eff_lexicon)/2)

  # Mutual information: I(S1, S2, R) = Hnorm(S1, S2) - Hnrom(S1, S2 | R)
  mi.append(mutual_information(result))

  count += 1


### Plot effective lexicon size and (normalized) mutual information ###

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].plot(gamma_values, l, '.-')
axs[0].set_xlabel("Gamma")
axs[0].set_ylabel("Effective Lexicon Size")

axs[1].plot(gamma_values, mi, '.-')
axs[1].set_xlabel("Gamma")
axs[1].set_ylabel("Normalized MI")

plt.savefig("figures/ext_lexicon_and_MI.png")
plt.clf()