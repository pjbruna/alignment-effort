from math import comb
import matplotlib.pyplot as plt
from models import *

n = 36
r = 0.5 # SDIC?
v = 2 / comb(n, 2)
t = 2 * (n**2)
l = 0.4

### Run model ###

mod = SignalEvolution(size=n, density=r)
result = mod.run_to_equilibrium(prob=v, lam=l, stop=t, print_output=False)

cost_over_time = result[1]
cond_entropy_over_time = result[2]
signal_entropy_over_time = result[3]

# Remove repetitions
cost_over_time = cost_over_time[:-t]
cond_entropy_over_time = cond_entropy_over_time[:-t]
signal_entropy_over_time = signal_entropy_over_time[:-t]

time = range(len(cost_over_time))

### Plot ###

fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(12, 12))

ax1.plot(time, cost_over_time, '.-')
ax1.set_xlabel("Time")
ax1.set_ylabel("Cost")
# ax1.set_ylim([0, 1])

ax2.plot(time, cond_entropy_over_time, '.-')
ax2.set_xlabel("Time")
ax2.set_ylabel("Conditional Entropy")
# ax2.set_ylim([0, 1])

ax3.plot(time, signal_entropy_over_time, '.-')
ax3.set_xlabel("Time")
ax3.set_ylabel("Signal Entropy")
# ax3.set_ylim([0, 1])

plt.show()

# plt.savefig("figures/rep_over_time.png")
plt.clf