import numpy as np
from math import comb
import matplotlib.pyplot as plt
from models import *

#### REPLICATION ####

# n = 36
# r = 0.5 # SDIC?
# v = 2 / comb(n, 2)
# t = 2 * (n**2)
# l = 0.4
# 
# ### Run model ###
# 
# mod = SignalEvolution(size=n, density=r)
# result = mod.run_to_equilibrium(prob=v, lam=l, stop=t, print_output=False)
# 
# cost_over_time = result[1]
# cond_entropy_over_time = result[2]
# signal_entropy_over_time = result[3]
# 
# # Remove repetitions
# cost_over_time = cost_over_time[:-t]
# cond_entropy_over_time = cond_entropy_over_time[:-t]
# signal_entropy_over_time = signal_entropy_over_time[:-t]
# 
# time = range(len(cost_over_time))
# 
# ### Plot ###
# 
# fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(12, 12))
# 
# ax1.plot(time, cost_over_time, '.-')
# ax1.set_xlabel("Time")
# ax1.set_ylabel("Cost")
# # ax1.set_ylim([0, 1])
# 
# ax2.plot(time, cond_entropy_over_time, '.-')
# ax2.set_xlabel("Time")
# ax2.set_ylabel("Conditional Entropy")
# # ax2.set_ylim([0, 1])
# 
# ax3.plot(time, signal_entropy_over_time, '.-')
# ax3.set_xlabel("Time")
# ax3.set_ylabel("Signal Entropy")
# # ax3.set_ylim([0, 1])
# 
# plt.savefig("figures/rep_over_time.png")
# plt.clf()
# 

#### EXTENSION ####

n = 6
m = n**2
r = 0.5 # SDIC?
v = 2 / comb(m, 2)
t = 2 * (m**2)
l = 0.4

### Run model ###

runs = 5

cost_over_time = []
cond_entropy_over_time = []
signal_entropy_over_time = []
jsd_over_time = []

for _ in range(runs):
    mod = JointSpeakerAlignment(signal=n, referent=m, density=r)
    result = mod.run_to_equilibrium(prob=v, lam=l, stop=t, print_output=False)

    cost_over_time.append(result[1][:-t])
    cond_entropy_over_time.append(result[2][:-t])
    signal_entropy_over_time.append(result[3][:-t])
    jsd_over_time.append(result[4][:-t])

### Plot ###
    
fig, axs = plt.subplots(4, 1, figsize=(10, 12))

# Plot results for each metric
for i, (metric, ylabel) in enumerate(zip([cost_over_time, cond_entropy_over_time, signal_entropy_over_time, jsd_over_time],
                                         ['Cost', 'Conditional Entropy', 'Signal Entropy', 'JSD'])):
    for j in range(runs):
        axs[i].plot(range(len(metric[j])), metric[j], label=f"Run {j+1}")
    
    # Add labels and legend for each subplot
    axs[i].set_xlabel('Time')
    axs[i].set_ylabel(ylabel)
    # axs[i].legend()
    # axs[i].grid(True)

plt.tight_layout()
plt.show()

# plt.savefig("figures/ext_over_time_5_runs_joint_sig.png")
plt.clf()

