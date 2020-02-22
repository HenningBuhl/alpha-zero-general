import os
import shutil
import time
import random
import numpy as np
import math
import sys
sys.path.append('..')
from utils import *
from pytorch_classification.utils import Bar, AverageMeter
from NeuralNet import NeuralNet

from .TicTacToeNNet import TicTacToeNNet as nnet

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.multiprocessing as mp


class NNetWrapper(NeuralNet):
    def __init__(self, game, args):
        super(NNetWrapper, self).__init__(game, args)
        self.device = torch.device('cuda' if self.args.cuda else 'cpu')
        self.nnet = nnet(game, self.args).to(self.device)
        self.optimizer = optim.Adam(self.nnet.parameters(),
                                    lr=self.args.lr,
                                    weight_decay=self.args.weight_decay if self.args.weight_decay is not None else 0)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.history = {'loss': [],
                        'pi_loss': [],
                        'v_loss': [],}


    def train(self, examples, verbose=1):
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = torch.FloatTensor(input_boards).to(self.device)
        target_pis = torch.FloatTensor(target_pis).to(self.device)
        target_vs = torch.FloatTensor(target_vs).to(self.device)
        dataset = TensorDataset(input_boards, target_pis, target_vs)
        params = {#'pin_memory': self.args.cuda, # Data is already on GPU.
                  'batch_size': self.args.batch_size,
                  'shuffle': True,
                  'drop_last': True, # Drop non-full batch size batches (BatchNorm can't handle batch_size=1)
                  #'num_workers': 0, # Multiprocessing is done otherwise.
                  }
        data_loader = DataLoader(dataset, **params)

        if verbose:
            bar = Bar('Training'.ljust(10, ' '), max=self.args.epochs*len(data_loader))
        eps_time = AverageMeter()
        data_time = AverageMeter()
        batch_time = AverageMeter()
        total_losses = AverageMeter()
        pi_losses = AverageMeter()
        v_losses = AverageMeter()
        end = time.time()
        batch = 0

        self.nnet.train()
        for epoch in range(self.args.epochs):
            for batch_idx, (boards, pi_target, v_target) in enumerate(data_loader):
                batch += 1
                data_time.update(time.time() - end) # Measure data preparation time.
                
                # Compute output.
                pi, v = self.nnet(boards)
                
                # Compute losses.
                pi_loss = self.loss_pi(pi_target, pi)
                v_loss = self.loss_v(v_target, v)
                total_loss = pi_loss + v_loss
                
                # Record loss.
                pi_losses.update(pi_loss.item(), boards.size(0))
                v_losses.update(v_loss.item(), boards.size(0))
                total_losses.update(total_loss.item(), boards.size(0))
                
                # Compute gradient and do optimizer step.
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                # Measure elapsed time.
                batch_time.update(time.time() - end)
                end = time.time()
                
                # Record metrics.
                self.history['loss'].append(total_losses.avg)
                self.history['pi_loss'].append(pi_losses.avg)
                self.history['v_loss'].append(v_losses.avg)
                
                # Plot prgress.
                if verbose:
                                 #f'Epoch {epoch}/{self.args.epochs}'\
                                 #f'Batch ({batch_idx+1}/{len(data_loader)}) '\
                                 #f'Data: {data_time.avg:.2f}s | '\
                                 #f'Batch: {batch_time.avg:.2f}s | '\
                                 #f'loss: {total_losses.avg:.2f} | '\
                                 #f'Total: {bar.elapsed_td:} | '\
                    bar.suffix = f''\
                                 f'({batch}/{self.args.epochs*len(data_loader)}) '\
                                 f'Eps Time: {eps_time.avg:.2f}s | '\
                                 f'Total: {bar.elapsed_td:} | '\
                                 f'ETA: {bar.eta_td:} | '\
                                 f'Pi Loss: {pi_losses.avg:.2f} | '\
                                 f'V Loss: {v_losses.avg:.2f}'\
                                 f''
                    bar.next()
        if verbose:
            bar.finish()


    def predict(self, board):
        board = torch.FloatTensor(board).to(self.device)
        board = board.view(1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)
        return torch.exp(pi).cpu().numpy()[0], v.cpu().numpy()[0]


    def loss_pi(self, targets, outputs):
        return -torch.sum(targets*outputs)/targets.size()[0]


    def loss_v(self, targets, outputs):
        return torch.sum((targets-outputs.view(-1))**2)/targets.size()[0] # Without .view(-1) -> loss really high. Why?


    def predict_fast(self, board):
        raise Exception('Fast policy net not supported.')


    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            #print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            #print("Checkpoint Directory exists! ")
            pass
        torch.save({
            'state_dict' : self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise Exception(f'No model in path \'{filepath}\'')
        map_location = None if self.args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])

