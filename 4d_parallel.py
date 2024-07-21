from math import comb
import matplotlib.pyplot as plt
import pandas as pd
from models import *
import multiprocessing as mp
import netCDF4 as nc

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
    mat_over_time = []

    mod = ReferentialAlignment(signal=n, referent=m, density=r)
    result = mod.run_to_equilibrium(prob=v, lam=l, stop=t)

    cost_over_time.append(result[1][:-t])
    cond_entropy_over_time.append(result[2][:-t])
    signal_entropy_over_time.append(result[3][:-t])
    jsd_over_time.append(result[4][:-t])
    ref_align_mi.append(result[5][:-t])
    combined_entropy_over_time.append(result[6][:-t])
    sparsity_over_time.append(result[7][:-t])

    mat_over_time.append(result[8])

    return cost_over_time, cond_entropy_over_time, signal_entropy_over_time, jsd_over_time, ref_align_mi, combined_entropy_over_time, sparsity_over_time, mat_over_time

### RUN ###

if __name__ == "__main__":
    runs = 1

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
    run_num = []

    for i in range(runs):
        cost.append(results[i][0][0])
        cond_ent.append(results[i][1][0])
        sig_ent.append(results[i][2][0])
        sig_jsd.append(results[i][3][0])
        ref_mi.append(results[i][4][0])
        sum_ent.append(results[i][5][0])
        sparsity.append(results[i][6][0])
        run_num.extend(np.repeat(i+1, len(results[i][6][0])))

        mat_data = np.array(results[i][7][0])
        mat_df = pd.DataFrame(mat_data)
        mat_df.to_csv(f'data/matrix_run={i}_l={l}.csv', index=False, header=False)

    ## STORE DATA ##

    log_cost_over_time = [item for sublist in cost for item in sublist]
    log_cond_entropy_over_time = [item for sublist in cond_ent for item in sublist]
    log_signal_entropy_over_time = [item for sublist in sig_ent for item in sublist]
    log_jsd_over_time = [item for sublist in sig_jsd for item in sublist]
    log_ref_align_mi = [item for sublist in ref_mi for item in sublist]
    log_combined_entropy_over_time = [item for sublist in sum_ent for item in sublist]
    log_sparsity_over_time = [item for sublist in sparsity for item in sublist]

    log_s1_cost_over_time = [item[0] for item in log_cost_over_time]
    log_s2_cost_over_time = [item[1] for item in log_cost_over_time]
    log_s1_cond_entropy_over_time = [item[0] for item in log_cond_entropy_over_time]
    log_s2_cond_entropy_over_time = [item[1] for item in log_cond_entropy_over_time]
    log_signal_entropy_over_time = [item[0] for item in log_signal_entropy_over_time]
    log_ref_align_mi = [item[0] for item in log_ref_align_mi]

    data = {
        's1_cost_over_time': log_s1_cost_over_time,
        's2_cost_over_time': log_s2_cost_over_time,
        's1_cond_entropy_over_time': log_s1_cond_entropy_over_time,
        's2_cond_entropy_over_time': log_s2_cond_entropy_over_time,
        'signal_entropy_over_time': log_signal_entropy_over_time,
        'jsd_over_time': log_jsd_over_time,
        'ref_align_mi': log_ref_align_mi,
        'combined_entropy_over_time': log_combined_entropy_over_time,
        'sparsity_over_time': log_sparsity_over_time,
        'run': run_num
    }

    df = pd.DataFrame(data)
    df.to_csv(f'data/4d_n={n}_v={v}_l={l}.csv', index=False)

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