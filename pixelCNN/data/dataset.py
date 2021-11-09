from torch.utils import data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import matplotlib.pyplot as plt

def get_dataset(config):
    """ Downloads cifar 10 dataset from torchvision and loads data.

    Args:
        config (argparse.NameSpace): configuration for data set.

    Returns:
        [torch.utils.data.DataLoader]: iterator over training set.
        [torch.utils.data.DataLoader]: iterator over test set.
    """
    kwargs = {'num_workers': config.num_workers, 'pin_memory': config.pin_memory, 'drop_last': config.drop_last}
    ds_transforms = transforms.Compose([transforms.ToTensor()])
    
    if config.dataset == 'cifar10':
        train_loader = data.DataLoader(
            datasets.CIFAR10(config.data_dir, download=True, train=True, transform=ds_transforms),
            batch_size=config.batch_size, shuffle=True, **kwargs
        )
        test_loader = data.DataLoader(
            datasets.CIFAR10(config.data_dir, download=True, train=False, transform=ds_transforms),
            batch_size=config.batch_size, shuffle=False, **kwargs
        )
    elif config.dataset == 'mnist':
        train_loader = data.DataLoader(
            datasets.MNIST(config.data_dir, download=True, train=True, transform=ds_transforms),
            batch_size=config.batch_size, shuffle=True, **kwargs
        )
        test_loader = data.DataLoader(
            datasets.MNIST(config.data_dir, download=True, train=False, transform=ds_transforms),
            batch_size=config.batch_size, shuffle=False, **kwargs
        )
    else:
        raise ValueError('Data set not supported.')
    
    return train_loader, test_loader

if __name__ == "__main__":
    # Test the loader
    config = {
        'num_workers': 0,
        'pin_memory': True,
        'drop_last': True,
        'data_dir': './data',
        'batch_size': 32,
        'dataset': 'mnist'
    }
    
    train_loader, test_loader = get_dataset(argparse.Namespace(**config))
    
    for x,y in train_loader:
        plot_image = x[0, :, :, :].permute(1, 2, 0)
        print("Image:", x)
        print("Label:", y)
        break
    plt.imshow(plot_image)
    plt.show()
    