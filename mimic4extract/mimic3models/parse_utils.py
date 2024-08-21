from __future__ import absolute_import
from __future__ import print_function

import re

def parse_task(log):
    """
    Determines the task type based on log content.

    :param log: String containing log data.
    :return: Task type ('multitask', 'los', 'decomp', 'pheno', 'ihm') or None if no match is found.
    """
    if re.search('ihm_C', log):
        return 'multitask'
    if re.search('partition', log):
        return 'los'
    if re.search('deep_supervision', log):
        return 'decomp'
    if re.search('ave_auc_micro', log):
        return 'pheno'
    if re.search('AUC of ROC', log):
        return 'ihm'
    return None

def get_loss(log, loss_name):
    """
    Extracts training and validation loss values from the log.

    :param log: String containing log data.
    :param loss_name: Name of the loss function (e.g., 'loss', 'ihm_loss', 'decomp_loss', 'pheno_loss', 'los_loss').
    :return: Tuple of lists containing training and validation loss values.
    """
    train = re.findall('[^_]{}: ([0-9.]+)'.format(loss_name), log)
    train = list(map(float, train))
    val = re.findall('val_{}: ([0-9.]+)'.format(loss_name), log)
    val = list(map(float, val))
    if len(train) > len(val):
        assert len(train) - 1 == len(val)
        train = train[:-1]
    return train, val

def parse_metrics(log, metric):
    """
    Extracts training and validation metric values from the log.

    :param log: String containing log data.
    :param metric: Name of the metric (e.g., 'accuracy', 'AUC').
    :return: Tuple of lists containing training and validation metric values.
    """
    ret = re.findall('{} = (.*)\n'.format(metric), log)
    ret = list(map(float, ret))
    if len(ret) % 2 == 1:
        ret = ret[:-1]
    return ret[::2], ret[1::2]

def parse_network(log):
    """
    Extracts the network type from the log.

    :param log: String containing log data.
    :return: Network type as a string.
    """
    ret = re.search("network='([^']*)'", log)
    return ret.group(1)

def parse_load_state(log):
    """
    Extracts the load state from the log.

    :param log: String containing log data.
    :return: Load state as a string.
    """
    ret = re.search("load_state='([^']*)'", log)
    return ret.group(1)

def parse_prefix(log):
    """
    Extracts the prefix from the log.

    :param log: String containing log data.
    :return: Prefix as a string.
    """
    ret = re.search("prefix='([^']*)'", log)
    return ret.group(1)

def parse_dim(log):
    """
    Extracts the dimension value from the log.

    :param log: String containing log data.
    :return: Dimension value as an integer.
    """
    ret = re.search("dim=([0-9]*)", log)
    return int(ret.group(1))

def parse_size_coef(log):
    """
    Extracts the size coefficient from the log.

    :param log: String containing log data.
    :return: Size coefficient as a string.
    """
    ret = re.search('size_coef=([\.0-9]*)', log)
    return ret.group(1)

def parse_depth(log):
    """
    Extracts the depth value from the log.

    :param log: String containing log data.
    :return: Depth value as an integer.
    """
    ret = re.search('depth=([0-9]*)', log)
    return int(ret.group(1))

def parse_ihm_C(log):
    """
    Extracts the IHM (In-Hospital Mortality) coefficient from the log.

    :param log: String containing log data.
    :return: IHM coefficient as a float or None if not found.
    """
    ret = re.search('ihm_C=([\.0-9]*)', log)
    if ret:
        return float(ret.group(1))
    return None

def parse_decomp_C(log):
    """
    Extracts the decompensation coefficient from the log.

    :param log: String containing log data.
    :return: Decompensation coefficient as a float or None if not found.
    """
    ret = re.search('decomp_C=([\.0-9]*)', log)
    if ret:
        return float(ret.group(1))
    return None

def parse_los_C(log):
    """
    Extracts the length of stay (LOS) coefficient from the log.

    :param log: String containing log data.
    :return: LOS coefficient as a float or None if not found.
    """
    ret = re.search('los_C=([\.0-9]*)', log)
    if ret:
        return float(ret.group(1))
    return None

def parse_pheno_C(log):
    """
    Extracts the phenotyping coefficient from the log.

    :param log: String containing log data.
    :return: Phenotyping coefficient as a float or None if not found.
    """
    ret = re.search('pheno_C=([\.0-9]*)', log)
    if ret:
        return float(ret.group(1))
    return None

def parse_dropout(log):
    """
    Extracts the dropout rate from the log.

    :param log: String containing log data.
    :return: Dropout rate as a float.
    """
    ret = re.search('dropout=([\.0-9]*)', log)
    return float(ret.group(1))

def parse_timestep(log):
    """
    Extracts the timestep value from the log.

    :param log: String containing log data.
    :return: Timestep value as a float.
    """
    ret = re.search('timestep=([\.0-9]*)', log)
    return float(ret.group(1))

def parse_partition(log):
    """
    Extracts the partition type from the log.

    :param log: String containing log data.
    :return: Partition type as a string or None if not found.
    """
    ret = re.search("partition='([^']*)'", log)
    if ret:
        return ret.group(1)
    return None

def parse_deep_supervision(log):
    """
    Extracts the deep supervision flag from the log.

    :param log: String containing log data.
    :return: Boolean indicating whether deep supervision is enabled.
    """
    ret = re.search('deep_supervision=(True|False)', log)
    if ret:
        return ret.group(1) == 'True'
    return False

def parse_target_repl_coef(log):
    """
    Extracts the target replacement coefficient from the log.

    :param log: String containing log data.
    :return: Target replacement coefficient as a float or None if not found.
    """
    ret = re.search('target_repl_coef=([\.0-9]*)', log)
    if ret:
        return float(ret.group(1))
    return None

def parse_epoch(state):
    """
    Extracts the epoch number from the state information.

    :param state: String containing state data.
    :return: Epoch number as an integer.
    """
    ret = re.search('.*(chunk|epoch)([0-9]*).*', state)
    return int(ret.group(2))

def parse_batch_size(log):
    """
    Extracts the batch size from the log.

    :param log: String containing log data.
    :return: Batch size as an integer.
    """
    ret = re.search('batch_size=([0-9]*)', log)
    return int(ret.group(1))

def parse_state(log, epoch):
    """
    Extracts the state file path corresponding to a specific epoch from the log.

    :param log: String containing log data.
    :param epoch: The epoch number to look for.
    :return: State file path as a string.
    :raises Exception: If the state file is not found.
    """
    lines = log.split('\n')
    for line in lines:
        res = re.search('.*saving model to (.*(chunk|epoch)([0-9]+).*)', line)
        if res is not None:
            if epoch == 0:
                return res.group(1).strip()
            epoch -= 1
    raise Exception("State file is not found")

def parse_last_state(log):
    """
    Extracts the last saved state file path from the log.

    :param log: String containing log data.
    :return: Last saved state file path as a string or None if not found.
    """
    lines = log.split('\n')
    ret = None
    for line in lines:
        res = re.search('.*saving model to (.*(chunk|epoch)([0-9]+).*)', line)
        if res is not None:
            ret = res.group(1).strip()
    return ret