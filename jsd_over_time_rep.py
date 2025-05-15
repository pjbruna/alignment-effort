from math import comb
import matplotlib.pyplot as plt
from models import *

n = 36
r = 0.5 # SDIC?
v = 2 / comb(n, 2)
t = 2 * (n**2)
l = 0.41

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

fig, ax = plt.subplots(figsize=(15, 10))

bin_size = 1

for i, (metric, ylabel, col) in enumerate(zip([cost_over_time, cond_entropy_over_time, signal_entropy_over_time],
                                                ['Weighted avg.', 'Hm(R|S)', 'Hn(S)'],
                                                ['black', 'gray', 'gray'])):
    run_off = len(metric) % bin_size
    series = metric[:-run_off] if run_off > 0 else metric
    series_binned = [np.mean(series[k:k+bin_size]) for k in range(0, len(series), bin_size)]
    ax.plot(range(len(series_binned)), series_binned, color=col)
    ax.annotate(ylabel, xy=(range(len(series_binned))[-1], series_binned[-1]), xytext=(range(len(series_binned))[-1] - 12, series_binned[-1] + 0.02), color='black', fontsize=15)

ax.set_xlabel(f'Time (1 unit = {bin_size} timesteps)', fontsize=20)
ax.set_ylabel('Value', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)

plt.show()

# plt.savefig("figures/rep_over_time.png")
plt.clf