import numpy as np
from math import comb
import matplotlib.pyplot as plt
from models import *

n = 4
m = n**2
r = 0.5 # SDIC? 
v = 2 / comb(63, 2)
t = 2 * (m**2)
lambda_values = np.linspace(0.4, 1, 13) #(0, 1, 21)

### Run model ###
count = 0
lex = []
mi = []
for l in lambda_values:
  print(f'Run: #{count}')

  mod = ReferentialAlignment(signal=n, referent=m, density=r)
  result = mod.run_to_equilibrium(prob=v, lam=l, stop=t)

  s1_result = result[0].mean(axis=1)
  s2_result = result[0].mean(axis=0)

  # Effective lexicon size:
  s1_lexicon = s1_result.sum(axis=0)
  s1_lexicon[s1_lexicon > 0] = 1

  s2_lexicon = s2_result.sum(axis=0)
  s2_lexicon[s2_lexicon > 0] = 1

  lex.append([(np.sum(s1_lexicon)/n**2), (np.sum(s2_lexicon)/n**2)])

  # Mutual information: I((S1, S2), R) = Hnorm(S1, S2) - Hnrom(S1, S2 | R)
  mi.append([mutual_information(s1_result), mutual_information(s2_result)])

  # # Effective lexicon size:
  # lexicon = result[0].sum(axis=0).sum(axis=0)
  # lexicon[lexicon > 0] = 1
  # lex.append(np.sum(lexicon)/n**2)
# 
  # # Mutual information: I(S1, S2, R1, R2) = Hnorm(S1, S2) - Hnrom(S1, S2 | R1, R2)
  # s1 = result[0].mean(axis=1)
  # s2 = result[0].mean(axis=0)
# 
  # s1_mi = mutual_information(s1)
  # s2_mi = mutual_information(s2)
# 
  # mi.append(np.mean([s1_mi, s2_mi]))

  count += 1


### Plot effective lexicon size and (normalized) mutual information ###

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

for i, (metric, ylabel) in enumerate(zip([lex, mi], ['Effective Lexicon Size', 'MI'])):
  s1 = [item[0] for item in metric]
  s2 = [item[1] for item in metric]

  axs[i].plot(lambda_values, s1)
  axs[i].plot(lambda_values, s2, linestyle="dashed")
  axs[i].set_xlabel("Lambda")
  axs[i].set_ylabel(ylabel)

plt.show()

# plt.savefig("figures/4d_lexicon_and_MI.png")
plt.clf()
