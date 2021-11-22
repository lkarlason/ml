from data.dataset import get_dataset
from models.pixelCNN import PixelCNN
from models.ema import *
from utils.helpers import mix_logistic_loss, mix_logistic_loss_1d
from utils.samplers import *
from data.dataset import get_dataset

from functools import partial
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import numpy as np
import torch
import shutil
import os
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.utils as utils


class SmoothedTrainRunner(object):
    def __init__(self, args, config):
        super(SmoothedTrainRunner, self).__init__()
        self.config = config
        self.args = args
        
    def train(self):
        # Trains a smoothed pixelCNN++
        obs = (1, 28, 28) if self.config.dataset == 'mnist' else (3, 32, 32)
        input_channels = obs[0]
        train_loader, test_loader = get_dataset(self.config)
        
        model = PixelCNN(self.config)
        model = model.to(self.config.device)
        model = torch.nn.DataParallel(model)
        sample_model = partial(model, sample=True)
        
        rescaling_inv = lambda x: .5*x + .5
        rescaling = lambda x: (x-.5)*2.
        
        if self.config.dataset == 'mnist':
            loss_op = lambda real, fake: mix_logistic_loss_1d(real, fake)
            sample_op = lambda x: sample_from_discretized_mix_logistic_1d(x, sample_model, self.config.nr_logistic_mix)

        elif self.config.dataset == 'cifar10':
            loss_op = lambda real, fake: mix_logistic_loss(real, fake)
            sample_op = lambda x: sample_from_discretized_mix_logistic(x, sample_model, self.config.nr_logistic_mix)

        
        optimizer = optim.Adam(model.parameters(), lr=self.config.lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.config.lr_decay)
        
        ckpt_path = os.path.join(self.args.log, 'pixelcnn_ckpts')
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
            
        if self.args.resume_training:
            state_dict = torch.load(os.path.join(ckpt_path, 'checkpoint.pth'), map_location=self.config.device)
            model.load_state_dict(state_dict[0])
            optimizer.load_state_dict(state_dict[1])
            scheduler.load_state_dict(state_dict[2])
            if len(state_dict) > 3:
                epoch = state_dict[3]
            print('model parameters loaded')
    
        tb_path = os.path.join(self.args.log, 'tensorboard')
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)
        os.makedirs(tb_path)    
        tb_logger = SummaryWriter(log_dir=tb_path)

        def debug_sample(model, data):
            model.eval()
            if not str(self.config.device) == 'cpu':
                data = data.cuda()
            with torch.no_grad():
                for i in range(obs[1]):
                    for j in range(obs[2]):
                        data_v = data
                        out_sample = sample_op(data_v)
                        data[:, :, i, j] = out_sample.data[:, :, i, j]
                return data


        print('starting training', flush=True)
        writes = 0
        for epoch in range(self.config.max_epochs):
            train_loss = 0.
            model.train()
            for batch_idx, (input, _) in enumerate(train_loader):
                if not str(self.config.device) == 'cpu':
                    input = input.cuda(non_blocking=True)
                input = input + torch.randn_like(input) * self.config.noise
                output = model(input)
                loss = loss_op(input, output)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                    
                train_loss += loss.item()
                if (batch_idx + 1) % self.config.print_every == 0:
                    deno = self.config.print_every * self.config.batch_size * np.prod(obs) * np.log(2.)
                    train_loss = train_loss / deno
                    print('epoch: {}, batch: {}, loss : {:.4f}'.format(epoch, batch_idx, train_loss), flush=True)
                    tb_logger.add_scalar('loss', train_loss, global_step=writes)
                    train_loss = 0.
                    writes += 1
                    
                    
                # decrease learning rate
                scheduler.step()
                
            test_model = model
            test_model.eval()
            test_loss = 0.
            with torch.no_grad():
                    for batch_idx, (input_var, _) in enumerate(test_loader):
                        if not str(self.config.device) == 'cpu':
                            input_var = input_var.cuda(non_blocking=True)

                        input_var = input_var + torch.randn_like(input_var) * self.config.noise
                        output = test_model(input_var)
                        loss = loss_op(input_var, output)
                        test_loss += loss.item()
                        del loss, output

                    deno = batch_idx * self.config.batch_size * np.prod(obs) * np.log(2.)
                    test_loss = test_loss / deno
                    print('epoch: %s, test loss : %s' % (epoch, test_loss), flush=True)
                    tb_logger.add_scalar('test_loss', test_loss, global_step=writes)
                    
            if (epoch + 1) % self.config.save_interval == 0:
                state_dict = [
                    model.state_dict(),
                    optimizer.state_dict(),
                    scheduler.state_dict(),
                    epoch,
                ]

                if (epoch + 1) % (self.config.save_interval * 2) == 0:
                    torch.save(state_dict, os.path.join(ckpt_path, f'ckpt_epoch_{epoch}.pth'))
                torch.save(state_dict, os.path.join(ckpt_path, 'checkpoint.pth'))

            if epoch % 10 == 0:
                print('sampling...', flush=True)
                sample_t = debug_sample(test_model, input_var[:25])
                sample_t = rescaling_inv(sample_t)

                if not os.path.exists(os.path.join(self.args.log, 'images')):
                    os.makedirs(os.path.join(self.args.log, 'images'))
                utils.save_image(sample_t, os.path.join(self.args.log, 'images', f'sample_epoch_{epoch}.png'),
                                 nrow=5, padding=0)    
        