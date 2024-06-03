from math import comb
import matplotlib.pyplot as plt
from models import *
import multiprocessing as mp

n = 4
m = n**2
r = 0.5 # SDIC?
v = 2 / comb(63, 2)
t = 2 * (m**2)
l = 0.6

def run_model(index):
    print(f'Running model: {index}')

    cost_over_time = []
    cond_entropy_over_time = []
    signal_entropy_over_time = []
    jsd_over_time = []
    ref_align_mi = []
    combined_entropy_over_time = []
    sparsity_over_time = []

    mod = ReferentialAlignment(signal=n, referent=m, density=r)
    result = mod.run_to_equilibrium(prob=v, lam=l, stop=t)

    cost_over_time.append(result[1][:-t])
    cond_entropy_over_time.append(result[2][:-t])
    signal_entropy_over_time.append(result[3][:-t])
    jsd_over_time.append(result[4][:-t])
    ref_align_mi.append(result[5][:-t])
    combined_entropy_over_time.append(result[6][:-t])
    sparsity_over_time.append(result[7][:-t])

    return cost_over_time, cond_entropy_over_time, signal_entropy_over_time, jsd_over_time, ref_align_mi, combined_entropy_over_time, sparsity_over_time

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
    ref_mi = []
    sum_ent = []
    sparsity = []

    for i in range(runs):
        cost.append(results[i][0][0])
        cond_ent.append(results[i][1][0])
        sig_ent.append(results[i][2][0])
        sig_jsd.append(results[i][3][0])
        ref_mi.append(results[i][4][0])
        sum_ent.append(results[i][5][0])
        sparsity.append(results[i][6][0])

    ## PLOTS ##

    bin_size = 1

    # Plot model info

    fig, axs = plt.subplots(4, 1, figsize=(10, 12))


    # Plot results for each metric
    for i, (metric, ylabel) in enumerate(zip([cost, cond_ent, sig_ent, sparsity],
                                             ['Cost', 'H(R|S1,S2)', 'H(S1,S2)', 'Sparsity'])):
        if(i>=2):
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

        else:
            for j in range(runs):
                s1 = [item[0] for item in metric[j]]
                s2 = [item[1] for item in metric[j]]

                # Speaker 1
                run_off = len(s1) % bin_size
                series = s1[:-run_off] if run_off > 0 else s1
                series_binned = [np.mean(series[k:k+bin_size]) for k in range(0, len(series), bin_size)]
                axs[i].plot(range(len(series_binned)), series_binned, color='gray', alpha=0.5)

                # Speaker 2
                run_off = len(s2) % bin_size
                series = s2[:-run_off] if run_off > 0 else s2
                series_binned = [np.mean(series[k:k+bin_size]) for k in range(0, len(series), bin_size)]
                axs[i].plot(range(len(series_binned)), series_binned, color='gray', alpha=0.5, linestyle='dashed')

            # Create avg.
            shortest_run = min(len(run) for run in metric)
            truncated = [run[:shortest_run] for run in metric]

            s1 = [[sublist[0] for sublist in sublist_list] for sublist_list in truncated]
            s2 = [[sublist[1] for sublist in sublist_list] for sublist_list in truncated]

            avg_s1 = np.mean(s1, axis=0)
            avg_s2 = np.mean(s2, axis=0)

            # Speaker 1
            run_off = len(avg_s1) % bin_size
            avg_series = avg_s1[:-run_off] if run_off > 0 else avg_s1
            avg_series_binned = [np.mean(avg_series[k:k+bin_size]) for k in range(0, len(avg_series), bin_size)]
            axs[i].plot(range(len(avg_series_binned)), avg_series_binned, color='black', label=f'S1')

            # Speaker 2
            run_off = len(avg_s2) % bin_size
            avg_series = avg_s2[:-run_off] if run_off > 0 else avg_s2
            avg_series_binned = [np.mean(avg_series[k:k+bin_size]) for k in range(0, len(avg_series), bin_size)]
            axs[i].plot(range(len(avg_series_binned)), avg_series_binned, color='black', linestyle = 'dashed', label=f'S2')

        # Add labels and legend for each subplot
        axs[3].set_xlabel(f'Time (1 unit = {bin_size} timesteps)', fontsize=15)
        axs[i].set_ylabel(ylabel, fontsize=15)
        # axs[i].legend()
        # axs[i].grid(True)
        # axs[0].set_title('Minimizing joint cost function')

    plt.tight_layout()
    plt.savefig(f'figures/4d_parallel_n={n}_t={t}.png')
    plt.clf()

    # Plot effect trends

    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # Plot results for each metric
    for i, (metric, ylabel) in enumerate(zip([ref_mi, sum_ent, sig_jsd],
                                             ['MI(R1,R2)', 'H(S1+S2)', 'JSD(S1,S2)'])):
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
        axs[2].set_xlabel(f'Time (1 unit = {bin_size} timesteps)', fontsize=15)
        axs[i].set_ylabel(ylabel, fontsize=15)
        # axs[i].legend()
        # axs[i].grid(True)
        # axs[0].set_title('Minimizing joint cost function')

    plt.tight_layout()
    plt.savefig(f'figures/4d_parallel_trends_n={n}_t={t}.png')
    plt.clf()