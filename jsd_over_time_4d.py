from math import comb
import matplotlib.pyplot as plt
from models import *

n = 4
m = n
r = 0.5 # SDIC?
v = 2 / comb(m**2, 2)
t = 2 * ((m**2)**2)
l = 0.4

### Run model ###

runs = 1

cost_over_time = []
cond_entropy_over_time = []
signal_entropy_over_time = []
jsd_over_time = []

for _ in range(runs):
    mod = ReferentialAlignment(signal=n, referent=m, density=r)
    result = mod.run_to_equilibrium(prob=v, lam=l, stop=t, print_output=False)

    cost_over_time.append(result[1][:-t])
    cond_entropy_over_time.append(result[2][:-t])
    signal_entropy_over_time.append(result[3][:-t])
    jsd_over_time.append(result[4][:-t])

### Plot ###
    
fig, axs = plt.subplots(4, 1, figsize=(12, 12))

# Line coloring
colors = ['C1', 'C2', 'C3', 'C4', 'C5']

# Plot results for each metric
for i, (metric, ylabel) in enumerate(zip([cost_over_time, cond_entropy_over_time, signal_entropy_over_time, jsd_over_time],
                                         ['Cost', 'Conditional Entropy', 'Signal Entropy', 'JSD'])):
    if(i==3):
        for j in range(runs):
            axs[i].plot(range(len(metric[j])), metric[j], label=f"Run {j+1}", c=colors[j])
    else:
        for j in range(runs):
            s1 = [item[0] for item in metric[j]]
            s2 = [item[1] for item in metric[j]]

            # Speaker 1
            axs[i].plot(range(len(s1)), s1, label=f"Run {j+1} S1", c=colors[j])

            # Speaker 2
            axs[i].plot(range(len(s2)), s2, linestyle="dashed", label=f"Run {j+1} S2", c=colors[j])
    
    # Add labels and legend for each subplot
    # axs[0, 0].set_title('Speaker 1')
    # axs[0, 1].set_title('Speaker 2')
    # axs[-2, 0].set_xlabel('Time')
    # axs[-2, 1].set_xlabel('Time')
    axs[i].set_xlabel('Time')
    axs[i].set_ylabel(ylabel)
    # axs[i].legend()
    # axs[i].grid(True)
    axs[0].set_title('Minimizing joint cost function')

# # Add subplot
# ax = fig.add_subplot(4, 1, 4)
# 
# for i, (metric, ylabel) in enumerate(zip([c], ['JSD'])):
#     for j in range(runs):
#         ax.plot(range(len(metric[j])), metric[j], label=f"Run {j+1}")
# 
#     # Add labels
#     ax.set_xlabel('Time')
#     ax.set_ylabel(ylabel)

plt.tight_layout()
plt.show()

# plt.savefig("figures/4d_over_time_joint_cost.png")
plt.clf()

