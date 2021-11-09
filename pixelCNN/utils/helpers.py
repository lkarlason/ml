import yaml
import os
import argparse

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