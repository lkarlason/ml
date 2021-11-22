import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import argparse
import os
import yaml

from data.dataset import get_dataset
from models.pixelCNN import PixelCNN

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def set_requires_grad(model):
    ## Disable gradients of pretrained model.
    for param in model.parameters():
        param.requires_grad = False

def main(config):
    ## Load data set.
    train_loader, test_loader = get_dataset(config)
    
    ## Load saved model.
    model = PixelCNN(config)
    config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    state_dict = torch.load(config.load_file, map_location=config.device)
    model.load_state_dict(state_dict)
    print('Model parameters loaded.')    
    set_requires_grad(model)
    
    ## Train with linear classifier on top
    k = config.probe_layer
    model.eval()
    out_features = 10
    if config.dataset == 'mnist':
        channels = 1
        if k == 1 or k == 4:
            dim = 14
        elif k == 2 or k == 3:
            dim = 7
        else:
            dim = 28
    else:
        channels = 3
        if k == 1 or k == 4:
            dim = 16
        elif k == 2 or k == 3:
            dim = 8
        else:
            dim = 32
    in_features = channels*dim*dim*config.nr_filters

    lin_model = torch.nn.Linear(in_features, out_features, device=config.device)
    optimizer = optim.Adam(lin_model.parameters(), lr=config.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=config.lr_decay)
    loss = torch.nn.CrossEntropyLoss()
    
    for epoch in range(config.max_epochs):
        train_loss = 0.
        for idx, (input, label) in enumerate(train_loader):
                if not str(config.device) == 'cpu':
                    input = input.cuda(non_blocking=True)
                input = input + torch.randn_like(input) * config.noise
                embedding = model.probe(input, k).contiguous().view(config.batch_size, in_features)
                output = lin_model(embedding)
                
                train_loss += loss(output, label).item()
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                # scheduler.step()
                break
        break 
    
    ## Evaluate on test set
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_file', type=str, default=os.path.join('saved_model', 'model.pth'))
    parser.add_argument('--config_file', type=str, default=os.path.join('configs', 'probe_mnist.yml'))
    
    args = parser.parse_args()
    with open(args.config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2namespace(config)
    config.load_file = args.load_file
    main(config)