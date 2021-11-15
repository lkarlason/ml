from data.dataset import get_dataset
from model.pixelCNN import PixelCNN

import torch
import functools


class SmoothedTrainRunner(object):
    def __init__(self, args, config):
        super(SmoothedTrainRunner, self).__init__()
        self.config = config
        self.args = args
        
    def train(self):
        # Trains a smoothed pixelCNN++
        obs = (1, 28, 28) if self.config.data.dataset == 'mnist' else (3, 32, 32)
        input_channels = obs[0]
        train_loader, test_loader = get_dataset(self.config.data)
        
        model = PixelCNN(self.config.model)
        model = model.to(self.config.device)
        model = torch.nn.DataParallel(model)
        sample_model = functools.partial(model, sample=True)
        
        rescaling_inv = lambda x: .5*x + .5
        rescaling = lambda x: (x-.5)*2.
        
        for x, y in train_loader:
            out = model(x)
            print(out.size())
            break        
