import numpy as np
import json

def save_stats(stats_dict, filename, capture=None):
    if len(filename) == 0:
        return
    def round_floats_list(lst, n=3):
        return [round(elem, n) for elem in lst]
    stats = {}
    stats['cmc_iters'] = round_floats_list(list(stats_dict['cmc'][0]))
    stats['cmc_vals'] = round_floats_list(list(stats_dict['cmc'][1]))
    stats['calc_counters'] = round_floats_list(list(stats_dict['avg_calcs']))
    stats['calc_counters_std'] = round_floats_list(list(stats_dict['std_dev_calcs']))
    stats['num_solved'] = stats_dict['num_solved']
    if capture:
        stats['capture'] = capture
    if 'avg_time' in stats_dict.keys():
        stats['avg_time'] = round(stats_dict['avg_time'], 3)

    with open(filename, 'w') as outfile:
        json.dump(stats, outfile)

def load_stats(filename):
    stats = {}
    with open(filename, 'r') as f:
        stats = json.load(f)
    return stats

def compute_stats(calc_stats, solved_status, execution_times=None):
    assert len(calc_stats) > 0
    assert len(calc_stats) == len(solved_status)

    num_solved = np.count_nonzero(solved_status)
    if num_solved > 0:
        calc_stats = np.array(calc_stats)
        avg_calcs = np.mean(calc_stats, axis = 0)
        std_dev_calcs = np.std(calc_stats, axis = 0)

        n_iters = calc_stats[:,0]
        n_iters_solved = n_iters[np.where(solved_status)]
        max_iters = np.max(n_iters)
        num_steps = 300
        check_points = np.linspace(1, max_iters + 1, endpoint=True, num=num_steps, dtype=np.float)
        cmc_values = []

        for i in range(len(check_points)):
            cmc_values.append(len(n_iters_solved[n_iters_solved < check_points[i]]))

        cmc_values = np.array(cmc_values) / float(len(solved_status))
    else:
        check_points=[0.]
        cmc_values=[0.]
        avg_calcs = [0.]*len(calc_stats[0])
        std_dev_calcs = [0.]*len(calc_stats[0])

    stats = {'cmc': (check_points, cmc_values), 'avg_calcs': avg_calcs, \
            'std_dev_calcs': std_dev_calcs,  'num_solved': num_solved}

    if execution_times != None:
        assert len(execution_times) == len(solved_status)
        avg_time = np.mean(execution_times)
        stats['avg_time'] = avg_time

    return stats
