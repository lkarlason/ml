from data.dataset import get_dataset
from utils.helpers import parse_args

import torch


CONFIG_FILE = 'config.yml'

def main(config):
    # Load data
    train_loader, test_loader = get_dataset(config.data)
    
    if config.data.dataset == 'mnist':
        channels = 1
        size = 28
    else:
        channels = 3
        size = 32
    
    # Check devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device " + str(device))

    # Create model
    

if __name__ == "__main__":
    # Parse yaml config file
    config = parse_args(CONFIG_FILE)
    
    main(config)