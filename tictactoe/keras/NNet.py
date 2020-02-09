import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import sys
sys.path.append('..')
from utils import *
from NeuralNet import NeuralNet

import argparse
from .TicTacToeNNet import TicTacToeNNet as onnet

"""
NeuralNet wrapper class for the TicTacToeNNet.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on (copy-pasted from) the NNet by SourKream and Surag Nair.
"""

class NNetWrapper(NeuralNet):
    def __init__(self, game, args):
        self.args = args
        self.nnet = onnet(game, self.args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.history = {}


    def train(self, examples):
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        
        if self.args.rollout == 'fast':
            # Train the combined model (pi, v, pi_fast).
            history = self.nnet.combined_model.fit(
                x=input_boards,
                y=[target_pis, target_vs, target_pis],
                batch_size=self.args.batch_size,
                epochs=self.args.epochs,
                verbose=1)
        else:
            # Train the model (pi, v).
            history = self.nnet.model.fit(
                x=input_boards,
                y=[target_pis, target_vs],
                batch_size=self.args.batch_size,
                epochs=self.args.epochs,
                verbose=1)
        
        # Add trainig results to history.
        for key in history.history.keys():
            if key not in self.history: # Add empty list to dict when key not known.
                self.history[key] = []
            self.history[key].extend(history.history[key])


    def predict(self, board):
        board = board[np.newaxis, :, :]
        pi, v = self.nnet.model.predict(board)
        return pi[0], v[0]


    def predict_fast(self, board):
        board = board[np.newaxis, :, :]
        pi_fast = self.nnet.fast_model.predict(board)
        return pi_fast[0]


    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            #print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            #print("Checkpoint Directory exists! ")
            pass
        self.nnet.model.save_weights(filepath)


    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path '{}'".format(filepath))
        self.nnet.model.load_weights(filepath)

