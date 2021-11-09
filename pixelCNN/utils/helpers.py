import yaml
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

def parse_args(config_file):
    """ Loads configuration from yaml file.

    Args:
        config_file (str): name of config file.

    Returns:
        [argparse.NameSpace]: configuration for model.
        [argparse.NameSpace]: configuration for data.
    """
    with open(os.path.join('configs', config_file), 'r') as f:
        config = yaml.load(f, Loader=Loader)
    
    for key, value in config.items():
        config[key] = argparse.Namespace(**value)
    ns = argparse.Namespace(**config)
    
    return ns


def down_shift(x, pad=None):
    # Removes last row
    xs = [int(y) for y in x.size()]
    x = x[:, :, :xs[2]-1, :]
    pad = nn.ZeroPad2d((0,0,1,0)) if pad is None else pad
    return pad(x)

def right_shift(x, pad=None):
    # Removes last column
    xs = [int(y) for y in x.size()]
    x = x[:, :, :, :xs[3]-1]
    pad = nn.ZeroPad2d((1,0,0,0)) if pad is None else pad
    return pad(x)

def concat_elu(x):
    # Concatenated exponential linear unit.
    axis = len(x.size()) - 3
    return F.elu(torch.cat([x, -x], dim=axis))