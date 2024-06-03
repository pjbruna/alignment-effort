from math import comb
import matplotlib.pyplot as plt
from models import *
import multiprocessing as mp

n = 6
m = n**2
r = 0.5 # SDIC?
v = 2 / comb(m, 2)
t = 2 * (m**2)
l = 0.41

def run_model(index):
    print(f'Running model: {index}')

    cost_over_time = []
    cond_entropy_over_time = []
    signal_entropy_over_time = []
    jsd_over_time = []
    combined_entropy_over_time = []
    sparsity_over_time = []

    mod = JointSpeakerAlignment(signal=n, referent=m, density=r)
    result = mod.run_to_equilibrium(prob=v, lam=l, stop=t, print_output=False)

    cost_over_time.append(result[1][:-t])
    cond_entropy_over_time.append(result[2][:-t])
    signal_entropy_over_time.append(result[3][:-t])
    jsd_over_time.append(result[4][:-t])
    combined_entropy_over_time.append(result[5][:-t])
    sparsity_over_time.append(result[6][:-t])

    return cost_over_time, cond_entropy_over_time, signal_entropy_over_time, jsd_over_time, combined_entropy_over_time, sparsity_over_time

### RUN ###

if __name__ == "__main__":
    runs = 2

    # Create a multiprocessing pool
    with mp.Pool() as pool:        
        results = pool.map(run_model, range(runs))
    
    # Compile results
    cost = []
    cond_ent = []
    sig_ent = []
    sig_jsd = []
    sum_ent = []
    sparsity = []

    for i in range(runs):
        cost.append(results[i][0][0])
        cond_ent.append(results[i][1][0])
        sig_ent.append(results[i][2][0])
        sig_jsd.append(results[i][3][0])
        sum_ent.append(results[i][4][0])
        sparsity.append(results[i][5][0])

    ## PLOTS ##

    bin_size = 1

    # Plot model info

    fig, axs = plt.subplots(4, 1, figsize=(10, 12))
    
    # Plot results for each metric
    for i, (metric, ylabel) in enumerate(zip([cost, cond_ent, sig_ent, sparsity],
                                             ['Cost', 'H(R|S1,S2)', 'H(S1,S2)', 'Sparsity'])):
        for j in range(runs):
            run_off = len(metric[j]) % bin_size
            series = metric[j][:-run_off] if run_off > 0 else metric[j]
            series_binned = [np.mean(series[k:k+bin_size]) for k in range(0, len(series), bin_size)]
            axs[i].plot(range(len(series_binned)), series_binned, color='gray', alpha=0.5)

        # Create avg.
        shortest_run = min(len(run) for run in metric)
        truncated = [run[:shortest_run] for run in metric]
        avg_metric = np.mean(truncated, axis=0)

        run_off = len(avg_metric) % bin_size
        avg_series = avg_metric[:-run_off] if run_off > 0 else avg_metric
        avg_series_binned = [np.mean(avg_series[k:k+bin_size]) for k in range(0, len(avg_series), bin_size)]
        axs[i].plot(range(len(avg_series_binned)), avg_series_binned, color='black')

        # Add labels and legend for each subplot
        axs[3].set_xlabel(f'Time (1 unit = {bin_size} timesteps)', fontsize=15)
        axs[i].set_ylabel(ylabel, fontsize=15)
        # axs[i].grid(True)

    # axs[0].set_title('Minimizing joint cost function')

    plt.tight_layout()
    plt.savefig(f'figures/3d_parallel_n={n}_t={t}.png')
    plt.clf()

    ### Figure ###

    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    # Plot results for each metric
    for i, (metric, ylabel) in enumerate(zip([sum_ent, sig_jsd],
                                             ['H(S1+S2)', 'JSD(S1,S2)'])):
        for j in range(runs):
            run_off = len(metric[j]) % bin_size
            series = metric[j][:-run_off] if run_off > 0 else metric[j]
            series_binned = [np.mean(series[k:k+bin_size]) for k in range(0, len(series), bin_size)]
            axs[i].plot(range(len(series_binned)), series_binned, color='gray', alpha=0.5)

        # Create avg.
        shortest_run = min(len(run) for run in metric)
        truncated = [run[:shortest_run] for run in metric]
        avg_metric = np.mean(truncated, axis=0)

        run_off = len(avg_metric) % bin_size
        avg_series = avg_metric[:-run_off] if run_off > 0 else avg_metric
        avg_series_binned = [np.mean(avg_series[k:k+bin_size]) for k in range(0, len(avg_series), bin_size)]
        axs[i].plot(range(len(avg_series_binned)), avg_series_binned, color='black')

        # Add labels and legend for each subplot
        axs[1].set_xlabel(f'Time (1 unit = {bin_size} timesteps)', fontsize=15)
        axs[i].set_ylabel(ylabel, fontsize=15)
        # axs[i].grid(True)
        # axs[0].set_title('Minimizing joint cost function')

    plt.tight_layout()
    plt.savefig(f'figures/3d_parallel_trends_n={n}_t={t}.png')
    plt.clf()