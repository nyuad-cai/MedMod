from __future__ import absolute_import
from __future__ import print_function

import argparse
from mimic3models import parse_utils
import json
import numpy as np

def check_decreasing(a, k, eps):
    """
    Check if the last `k` values of a list `a` are decreasing by at least `eps`.

    Parameters:
    - a (list): The list of values to check.
    - k (int): The number of last values to consider.
    - eps (float): The minimum decrease threshold.

    Returns:
    - bool: True if the values are decreasing by at least `eps`, otherwise False.
    """
    if k >= len(a):
        return False
    pos = len(a) - 1
    for i in range(k):
        if a[pos] > a[pos - 1] + eps:
            return False
        pos -= 1
    return True

def process_single(filename, verbose, select):
    """
    Process a single log file to determine if it needs to be rerun based on metrics.

    Parameters:
    - filename (str): The path to the log file.
    - verbose (bool): If True, print additional information during processing.
    - select (bool): If True, process the file; otherwise, always rerun.

    Returns:
    - dict or None: A dictionary with details if the file needs to be rerun, otherwise None.
    """
    if verbose:
        print(f"Processing log file: {filename}")

    with open(filename, 'r') as fin:
        log = fin.read()
    
    task = parse_utils.parse_task(log)

    if task is None:
        print(f"Task is not detected: {filename}")
        return None

    if verbose:
        print(f"\ttask = {task}")

    metric_map = {
        'multitask': 'ave_auc_macro',
        'pheno': 'ave_auc_macro',
        'ihm': 'AUC of ROC',
        'decomp': 'AUC of ROC',
        'los': 'Cohen kappa score'
    }
    metric = metric_map.get(task, None)
    if metric is None:
        raise ValueError("Unknown task type")

    train_metrics, val_metrics = parse_utils.parse_metrics(log, metric)
    if not train_metrics:
        print(f"Less than one epoch: {filename}")
        return None
    
    last_train = train_metrics[-1]
    last_val = val_metrics[-1]

    if verbose:
        print(f"\tlast train = {last_train}, last val = {last_val}")

    rerun = True
    task_conditions = {
        'ihm': [(0.83, 0.88), (0.84, 0.89), (0.85, 0.9)],
        'decomp': [(0.85, 0.89), (0.87, 0.9), (0.88, 0.92)],
        'pheno': [(0.75, 0.77), (0.76, 0.79)],
        'multitask': [(0.75, 0.77), (0.76, 0.79)],
        'los': [(0.35, 0.42), (0.38, 0.44)]
    }
    if task in task_conditions:
        for val_thresh, train_thresh in task_conditions[task]:
            if last_val < val_thresh and last_train > train_thresh:
                rerun = False
                break

    # Check if validation metrics are decreasing
    n_decreases = 3 if task in ['ihm', 'decomp', 'pheno', 'multitask'] else 5
    if check_decreasing(val_metrics, n_decreases, 0.001):
        rerun = False

    # Check if maximum validation value was very early
    tol = 0.01 if task in ['ihm', 'decomp', 'pheno', 'multitask'] else 0.03
    val_max = max(val_metrics)
    val_max_pos = np.argmax(val_metrics)
    if len(val_metrics) - val_max_pos >= 8 and val_max - last_val > tol:
        rerun = False

    if not select:
        rerun = True

    if verbose:
        print(f"\trerun = {rerun}")

    if not rerun:
        return None

    # Need to rerun
    last_state = parse_utils.parse_last_state(log)
    if last_state is None:
        print(f"Last state is not parsed: {filename}")
        return None

    n_epochs = parse_utils.parse_epoch(last_state)
    if verbose:
        print(f"\tlast state = {last_state}")

    network = parse_utils.parse_network(log)
    prefix = parse_utils.parse_prefix(log)
    if prefix == '':
        prefix = 'r2'
    elif not prefix[-1].isdigit():
        prefix += '2'
    else:
        prefix = prefix[:-1] + str(int(prefix[-1]) + 1)

    dim = parse_utils.parse_dim(log)
    size_coef = parse_utils.parse_size_coef(log)
    depth = parse_utils.parse_depth(log)

    ihm_C = parse_utils.parse_ihm_C(log)
    decomp_C = parse_utils.parse_decomp_C(log)
    los_C = parse_utils.parse_los_C(log)
    pheno_C = parse_utils.parse_pheno_C(log)

    dropout = parse_utils.parse_dropout(log)
    partition = parse_utils.parse_partition(log)
    deep_supervision = parse_utils.parse_deep_supervision(log)
    target_repl_coef = parse_utils.parse_target_repl_coef(log)

    batch_size = parse_utils.parse_batch_size(log)

    command = f"python -u main.py --network {network} --prefix {prefix} --dim {dim}"\
              f" --depth {depth} --epochs 100 --batch_size {batch_size} --timestep 1.0"\
              f" --load_state {last_state}"

    if 'channel' in network:
        command += f' --size_coef {size_coef}'

    if ihm_C is not None:
        command += f' --ihm_C {ihm_C}'

    if decomp_C is not None:
        command += f' --decomp_C {decomp_C}'

    if los_C is not None:
        command += f' --los_C {los_C}'

    if pheno_C is not None:
        command += f' --pheno_C {pheno_C}'

    if dropout > 0.0:
        command += f' --dropout {dropout}'

    if partition:
        command += f' --partition {partition}'

    if deep_supervision:
        command += ' --deep_supervision'

    if target_repl_coef is not None and target_repl_coef > 0.0:
        command += f' --target_repl_coef {target_repl_coef}'

    return {
        "command": command,
        "train_max": np.max(train_metrics),
        "train_max_pos": np.argmax(train_metrics),
        "val_max": np.max(val_metrics),
        "val_max_pos": np.argmax(val_metrics),
        "last_train": last_train,
        "last_val": last_val,
        "n_epochs": n_epochs,
        "filename": filename
    }

def main():
    """
    Main function to process multiple log files and generate rerun commands.
    """
    argparser = argparse.ArgumentParser(description="Process and generate rerun commands for log files.")
    argparser.add_argument('logs', type=str, nargs='+', help="Log files to process.")
    argparser.add_argument('--verbose', type=int, default=0, help="Set verbosity level.")
    argparser.add_argument('--select', dest='select', action='store_true', help="If set, process files; otherwise always rerun.")
    argparser.add_argument('--no-select', dest='select', action='store_false', help="If set, always rerun files.")
    argparser.set_defaults(select=True)
    args = argparser.parse_args()

    if not isinstance(args.logs, list):
        args.logs = [args.logs]

    rerun = []
    for log in args.logs:
        if ".log" not in log:  # Not a log file or is a not renamed log file
            continue
        ret = process_single(log, args.verbose, args.select)
        if ret:
            rerun.append(ret)
    
    rerun = sorted(rerun, key=lambda x: x["last_val"], reverse=True)

    print(f"Need to rerun {len(rerun)} / {len(args.logs)} models")
    print("Saving the results in rerun_output.json")
    with open("rerun_output.json", 'w') as fout:
        json.dump(rerun, fout, indent=4)

    print("Saving commands in rerun_commands.sh")
    with open("rerun_commands.sh", 'w') as fout:
        for item in rerun:
            fout.write(item['command'] + '\n')

    print("Saving filenames in rerun_filenames.txt")
    with open("rerun_filenames.txt", 'w') as fout:
        for item in rerun:
            fout.write(item['filename'] + '\n')

if __name__ == '__main__':
    main()