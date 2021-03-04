import numpy as np
import torch
import pickle


def initialize():
    """
    Initialize dictionary with stats for logging.

    Returns:
    logs    -- Dictionary with defined keys
    """
    stats = ["inference_time", "num_objects", "min_conf"]
    logs = {new_list: [] for new_list in stats}

    return logs


def writeLog(logs, time_begin, time_end, labels, conf):
    """
    Writes log info into dictionary.

    Args:
    logs        -- Dictionary with log info
    time_begin  -- Begin of inference time for current frame
    time_end    -- End of inference time for current frame
    labels      -- Torch tensor with object label indices
    conf        -- Torch tensor with detection confidences

    Return:
    logs        -- Updated dictionary with log information
    """
    logs["inference_time"].append(time_end - time_begin)
    if (len(labels.numpy() > 0)):
        logs["num_objects"].append(len(labels.numpy()))
        logs["min_conf"].append(min(conf.numpy()))
    else:
        logs["num_objects"].append(None)
        logs["min_conf"].append(None)

    return logs


# TODO: Handle case of folder not available
# TODO: Add number or other naming system to logfiles
def saveLogs(logs):
    """
    Saves logs to disk.

    Args:
    logs    -- Dictionary with log data
    
    """
    path_out = "dev/logs"
    name = "logfile"

    with open(path_out + "/" + name + ".log", 'wb') as handle:
        pickle.dump(logs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Logs saved as " + name +  " to path " + "dev/logs")