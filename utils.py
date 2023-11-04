import os
from tqdm import tqdm
import numpy as np

import torch as T

USE_CUDA = T.cuda.is_available()

def prRed(prt):
    print("\033[91m{}\033[00m" .format(prt))


def prGreen(prt):
    print("\033[92m{}\033[00m" .format(prt))


def prYellow(prt):
    print("\033[93m{}\033[00m" .format(prt))


def prLightPurple(prt):
    print("\033[94m{}\033[00m" .format(prt))


def prPurple(prt):
    print("\033[95m{}\033[00m" .format(prt))


def prCyan(prt):
    print("\033[96m{}\033[00m" .format(prt))


def prLightGray(prt):
    print("\033[97m{}\033[00m" .format(prt))


def prBlack(prt):
    print("\033[98m{}\033[00m" .format(prt))


def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()

def to_tensor(x, dtype="float"):
    """
    Numpy array to tensor
    """
    
    FloatTensor = T.cuda.FloatTensor if USE_CUDA else T.FloatTensor
    LongTensor = T.cuda.LongTensor if USE_CUDA else T.LongTensor
    ByteTensor = T.cuda.ByteTensor if USE_CUDA else T.ByteTensor

    if dtype == "float":
        x = np.array(x, dtype=np.float64).tolist()
        return FloatTensor(x)
    elif dtype == "long":
        x = np.array(x, dtype=np.long).tolist()
        return LongTensor(x)
    elif dtype == "byte":
        x = np.array(x, dtype=np.byte).tolist()
        return ByteTensor(x)
    else:
        x = np.array(x, dtype=np.float64).tolist()

    return FloatTensor(x)

def soft_update(target, source, tau):
    """
    Performs a soft target update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    """
    Performs a hard target update
    """
    for target_param, param in zip(target_param.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def get_output_folder(parent_dir, env_name):
    """
    Return save folder

    Assumes folders in the parent_dir have suffix -run{run 
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results 
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
        Path of the directory containing all experiment runs.
    
    Returns
    -------
    parent_dir/run_dir
        Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for file_or_dir in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, file_or_dir)):
            continue
        try:
            folder_name = int(file_or_dir.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1
    
    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + f'-run{experiment_id}'
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir

