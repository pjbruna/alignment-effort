from math import comb
import matplotlib.pyplot as plt
from models import *

n = 4
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

axs[0].set_title('Minimizing joint cost function')

plt.tight_layout()
# plt.show()

plt.savefig("figures/ext_over_time_5_runs_joint_cost.png")
plt.clf()

