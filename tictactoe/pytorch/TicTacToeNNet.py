import sys
sys.path.append('..')
from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

class TicTacToeNNet(nn.Module):
    def __init__(self, game, args):
        # Game params.
        self.game = game
        self.board_x, self.board_y = self.game.getBoardSize()
        self.action_size = self.game.getActionSize()
        self.args = args
        
        if self.args.rollout == 'fast':
            raise Exception('Fast policy net not supported.')

        if self.args.useCustomInput:
            customInputShape = self.game.getCustomInputShape()
            self.input_channels = customInputShape[0]
        else:
            self.input_channels = 1
        
        super(TicTacToeNNet, self).__init__()
        
        self.conv = nn.Conv2d(in_channels=self.input_channels,
                        out_channels=self.args.num_channels,
                        kernel_size=self.args.kernel_size,
                        stride=1,
                        padding=self.args.kernel_size//2)
        
        if self.args.dual:
            self.body = Body(self.game, self.args)
            self.head = Head(self.game, self.args)
            self.pi = Pi(self.game, self.args)
            self.v = V(self.game, self.args)
        else:
            self.pi_body = Body(self.game, self.args)
            self.pi_head = Head(self.game, self.args)
            self.pi = Pi(self.game, self.args)

            self.v_body = Body(self.game, self.args)
            self.v_head = Head(self.game, self.args)
            self.v = V(self.game, self.args)

    def forward(self, s):
        s = s.view(-1, self.input_channels, self.board_x, self.board_y) # Add channel dimension.
        if self.args.dual:
            s = self.conv(s)
            s = self.body(s)
            s = self.head(s)
            pi = self.pi(s)
            v = self.v(s)
        else:
            pi = self.conv(s)
            pi = self.pi_body(s)
            pi = self.pi_head(pi)
            pi = self.pi(pi)

            v = self.conv(s)
            v = self.v_body(s)
            v = self.v_head(v)
            v = self.v(v)
        return F.log_softmax(pi, dim=1), torch.tanh(v)


class Block(nn.Module):
    def __init__(self, game, args, rank):
        self.game = game
        self.args = args
        self.rank = rank
        super(Block, self).__init__()
        self.left = self.get_left()
        self.right = self.get_right()
    
    def get_left(self):
        layers = []
        for repeat in range(self.args.block_repeats):
            double_channel = repeat == 0 and self.rank != 0 and self.args.residual == 'concat' # Previous block had concat (cat) layer.
            num_channels = self.args.num_channels*2 if double_channel else self.args.num_channels
            layers.append(nn.Conv2d(num_channels,
                                    self.args.num_channels,
                                    self.args.kernel_size,
                                    stride=1,
                                    padding=self.args.kernel_size//2))
            if self.args.useBatchNorm:
                layers.append(nn.BatchNorm2d(self.args.num_channels))
            if repeat+1 < self.args.block_repeats: # Add activation if not last inner block.
                layers.append(nn.ReLU())
        return nn.ModuleList(layers)

    def get_right(self):
        layers = []
        if self.args.residual is not None:
            double_channel = self.rank != 0 and self.args.residual == 'concat' # Previous block had concat (cat) layer.
            num_channels = self.args.num_channels*2 if double_channel else self.args.num_channels
            layers.append(nn.Conv2d(num_channels,
                                    self.args.num_channels,
                                    self.args.kernel_size,
                                    stride=1,
                                    padding=self.args.kernel_size//2))
            if self.args.useBatchNorm:
                layers.append(nn.BatchNorm2d(self.args.num_channels))
        return nn.ModuleList(layers)
    

    def forward(self, s):
        s_left = s
        for layer in self.left:
            s_left = layer(s_left)
        
        if self.args.residual is not None:
            s_right = s
            for layer in self.right:
                s_right = layer(s_right)

            if self.args.residual == 'add':
                s = s_left + s_right
            elif self.args.residual == 'concat':
                s = torch.cat((s_left, s_right), 1)
            else:
                raise ValueError(f'Residual connection type \'{self.args.residual}\' is not supported.')
        else:
            s = s_left
        return F.relu(s)


class Body(nn.Module):
    def __init__(self, game, args):
        self.game = game
        self.args = args
        super(Body, self).__init__()
        self.blocks = nn.ModuleList([Block(self.game, self.args, i) for i in range(self.args.num_blocks)])

    def forward(self, s):
        for block in self.blocks:
            s = block(s)
        return s


class Head(nn.Module):
    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.input_num = self.args.num_channels * np.prod(game.getBoardSize())
        self.double_channel = False
        if self.args.residual == 'concat':
            self.input_num *= 2
            self.double_channel = True
        super(Head, self).__init__()
        if self.args.action2D:
            self.layers = self.get_conv()
        else:
            self.layers = self.get_dense()

    def get_dense(self):
        layers = []
        layers.append(nn.Flatten())
        for i in range(len(self.args.dense_layers)):
            layers.append(nn.Linear(self.input_num if i == 0 else self.args.dense_layers[i-1], self.args.dense_layers[i]))
            if self.args.useBatchNorm:
                layers.append(nn.BatchNorm1d(self.args.dense_layers[i]))
            layers.append(nn.ReLU())
            if self.args.dropout > 0.:
                layers.append(nn.Dropout(p=self.args.dropout))
        return nn.ModuleList(layers)

    def get_conv(self):
        layers = []
        for i in range(len(self.args.dense_layers)):
            num_channels = self.args.num_channels*2 if self.double_channel and i == 0 else self.args.num_channels
            layers.append(nn.Conv2d(num_channels,
                                    self.args.num_channels,
                                    self.args.kernel_size,
                                    stride=1,
                                    padding=self.args.kernel_size//2))
            if self.args.useBatchNorm:
                layers.append(nn.BatchNorm2d(self.args.num_channels))
            layers.append(nn.ReLU())
        return nn.ModuleList(layers)

    def forward(self, s):
        for layer in self.layers:
            s = layer(s)
        return s


class Pi(nn.Module):
    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.action_size = self.game.getActionSize()
        super(Pi, self).__init__()
        if self.args.action2D:
            self.layers = nn.ModuleList([nn.Conv2d(self.args.num_channels, 
                                                   1,
                                                   self.args.kernel_size,
                                                   stride=1,
                                                   padding=self.args.kernel_size//2),
                                         Reshape(-1, self.action_size)])
        else:
            self.layers = nn.ModuleList([nn.Linear(self.args.dense_layers[-1], self.action_size)])

    def forward(self, s):
        for layer in self.layers:
            s = layer(s)
        return s


class V(nn.Module):
    def __init__(self, game, args):
        self.game = game
        self.args = args
        if self.args.action2D:
            self.input_num = self.args.num_channels * self.args.kernel_size**2
        else:
            self.input_num = self.args.dense_layers[-1]
        super(V, self).__init__()
        if self.args.action2D:
            self.layers = nn.ModuleList([nn.Flatten(),
                                         nn.Linear(self.input_num, 1)])
        else:
            self.layers = nn.ModuleList([nn.Linear(self.input_num, 1)])

    def forward(self, s):
        for layer in self.layers:
            s = layer(s)
        return s


class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape)

