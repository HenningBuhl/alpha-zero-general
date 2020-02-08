import sys
sys.path.append('..')
from utils import *

import argparse
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.regularizers import *

"""
NeuralNet for the game of TicTacToe.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on the OthelloNNet by SourKream and Surag Nair.
"""

class TicTacToeNNet():
    def __init__(self, game, args):
        # Game params.
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Network parameters.
        self.reg = l2(self.args.weight_decay) if self.args.weight_decay else None

        # Input.
        self.input_boards = Input(shape=(self.board_x, self.board_y), name='Input_Board')
        x = Reshape((self.board_x, self.board_y, 1), name='Reshape')(self.input_boards)

        # Use dual or separate model.
        if self.args.dual:
            x = self.body(x, prefix='Body') # Body.
            x = self.head(x, prefix='Head') # Head.
            self.pi = self.head_pi(x) # pi.
            self.v = self.head_v(x) # v.
        else:
            x_pi = self.body(x, prefix='Body_pi') # Body pi.
            x_pi = self.head(x_pi, prefix='Head_pi') # Head pi.
            self.pi = self.head_pi(x_pi) # pi.
            x_v = self.body(x, prefix='Body_v') # Body v.
            x_v = self.head(x_v, prefix='Head_v') # Head v.
            self.v = self.head_v(x_v) # v.
        
        # Model.
        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v], name='TicTacToeNNet')
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(self.args.lr))
        
        if self.args.rollout == 'fast':
            # Fast policy net.
            x_fast = Flatten(name='Flatten_pi_fast')(self.input_boards)
            self.pi_fast = Dense(self.action_size, activation='softmax', name='pi_fast')(x_fast)
            self.fast_model = Model(inputs=self.input_boards, outputs=self.pi_fast, name='TicTacToeNNet_fast')
            self.fast_model.compile(loss='categorical_crossentropy', optimizer=Adam(self.args.lr))

            # Combined model (pi, v, pi_fast).
            self.combined_model = Model(inputs=self.input_boards, outputs=[self.pi, self.v, self.pi_fast], name='TicTacToeNNet_combined')
            self.combined_model.compile(loss=['categorical_crossentropy', 'mean_squared_error', 'categorical_crossentropy'], optimizer=Adam(self.args.lr))


    def body(self, x, prefix):
        for i in range(self.args.num_blocks):
            x = self.conv_block(x, i+1, prefix=prefix)
        return x


    def head(self, x, prefix):
        if self.args.action2D:
            for i in range(2):
                x = self.conv_block(x, i+1, prefix=prefix)
        else:
            x = Flatten(name=f'{prefix}_Flatten')(x)
            for i, n in zip(range(2), [2**9, 2**8]):
                x = self.dense_block(x, n, i, prefix=prefix)
        return x


    def conv_block(self, x, block_id, prefix):
        block_predix = f'{prefix}_{block_id}'
        x_residual = x
        
        for i in range(self.args.block_repeats):
            x = Conv2D(self.args.num_channels, 3, padding='same', kernel_regularizer=self.reg, name=f'{block_predix}_Conv2D_{i+1}')(x)
            x = BatchNormalization(name=f'{block_predix}_BatchNorm_{i+1}')(x)
            if i+1 == self.args.block_repeats and self.args.residual is not None:
                x_residual = Conv2D(self.args.num_channels, 3, padding='same', kernel_regularizer=self.reg, name=f'{block_predix}_Conv2D_Residual')(x_residual)
                x_residual = BatchNormalization(name=f'{block_predix}_BatchNorm_Residual')(x_residual)
                if self.args.residual == 'add':
                    x = Add(name=f'{block_predix}_Add')([x, x_residual])
                elif self.args.residual == 'concat':
                    x = Concatenate(name=f'{block_predix}_Concatenate')([x, x_residual])
                else:
                    raise ValueError(f'Residual connection type \'{self.args.residual}\' is not supported.')
            x = Activation('relu', name=f'{block_predix}_Activation_{i+1}')(x)
        return x


    def dense_block(self, x, n, i, prefix):
        head_prefix = f'{prefix}_{i+1}'
        x = Dense(n, kernel_regularizer=self.reg, name=f'{head_prefix}_Dense')(x)
        x = BatchNormalization(name=f'{head_prefix}_BatchNorm')(x)
        x = Activation('relu', name=f'{head_prefix}_Activation')(x)
        x = Dropout(self.args.dropout, name=f'{head_prefix}_Dropout')(x)
        return x


    def head_v(self, x):
        if self.args.action2D:
            x = Flatten(name=f'Flatten_v')(x)
            x = Dense(1, activation='tanh', kernel_regularizer=self.reg, name='v')(x)
        else:
            x = Dense(1, activation='tanh', kernel_regularizer=self.reg, name='v')(x)
        return x


    def head_pi(self, x):
        if self.args.action2D:
            x = Conv2D(1, (self.board_x, self.board_y), padding='same', activation='softmax', name='pi')(x)
            x = Reshape((self.action_size,), name='Reshape_pi')(x)
        else:
            x = Dense(self.action_size, activation='softmax', kernel_regularizer=self.reg, name='pi')(x)
        return x

