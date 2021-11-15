from data.dataset import get_dataset
from utils.helpers import parse_args
from model.pixelCNN import PixelCNN
from runners.SmoothedTrainRunner import SmoothedTrainRunner


import torch
import time

CONFIG_FILE = 'config_mnist.yml'

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
    config.device = device
    print("Using device " + str(device))

    # Run training
    runner = SmoothedTrainRunner(None, config)
    runner.train()
    

if __name__ == "__main__":
    # Parse yaml config file
    config = parse_args(CONFIG_FILE)
    
    main(config)