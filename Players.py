import numpy as np
from utils import *

"""
Random and Human-ineracting players for the game of TicTacToe.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on the OthelloPlayers by Surag Nair.

"""

class Player():
    def __init__(self, game, args=None, verbose=False):
        self.game = game
        self.args = args
        self.verbose = verbose

    def play(self, board, verbose=False):
        pass
    
    def get_verbose(self, verbose=None):
        return self.verbose if verbose is None else verbose

class RandomPlayer(Player):
    def __init__(self, game, args=None, verbose=False):
        super(RandomPlayer, self).__init__(game, args, verbose)

    def play(self, board, verbose=None):
        verbose = super(RandomPlayer, self).get_verbose(verbose)
        valids = self.game.getValidMoves(board, 1)
        valid_indices = [i for i, valid in enumerate(valids) if valid]
        a = np.random.choice(valid_indices)
        if verbose:
            print(f'Playing random available move: {a}')
        return a


class HumanPlayer(Player):
    def __init__(self, game, args=None, verbose=False):
        super(HumanPlayer, self).__init__(game, args, verbose)

    def play(self, board, verbose=None):
        verbose = super(HumanPlayer, self).get_verbose(verbose)
        valid_moves = self.game.getValidMoves(board, 1)
        uf_moves = self.game.getUserFriendlyMoves(board, 1)
        for i, (valid, uf_move) in enumerate(uf_moves):
            if valid:
                if self.verbose:
                    print(f'Move: {i:2d}, UF: {uf_move}')

        while True:
            move = int(input())
            if valid_moves[move]:
                break
            else:
                if verbose:
                    print('Invalid move')
        if verbose:
            print(f'Playing action {move}')
        return move


class OneStepLookaheadPlayer(Player):
    """Simple player who always takes a win if presented, or blocks a loss if obvious, otherwise is random."""
    def __init__(self, game, args=None, verbose=False):
        super(OneStepLookaheadPlayer, self).__init__(game, args, verbose)
        self.player_num = 1

    def play(self, board, verbose=None):
        verbose = super(OneStepLookaheadPlayer, self).get_verbose(verbose)
        valid_moves = self.game.getValidMoves(board, self.player_num)
        win_move_set = set()
        fallback_move_set = set()
        stop_loss_move_set = set()
        for move, valid in enumerate(valid_moves):
            if not valid: continue
            if -self.player_num == self.game.getGameEnded(*self.game.getNextState(board, self.player_num, move)):
                win_move_set.add(move)
            elif -self.player_num == self.game.getGameEnded(*self.game.getNextState(board, -self.player_num, move)):
                stop_loss_move_set.add(move)
            else:
                fallback_move_set.add(move)

        if len(win_move_set) > 0:
            ret_move = np.random.choice(list(win_move_set))
            if verbose:
                print('Playing winning action %s from %s' % (ret_move, win_move_set))
        elif len(stop_loss_move_set) > 0:
            ret_move = np.random.choice(list(stop_loss_move_set))
            if verbose:
                print('Playing loss stopping action %s from %s' % (ret_move, stop_loss_move_set))
        elif len(fallback_move_set) > 0:
            ret_move = np.random.choice(list(fallback_move_set))
            if verbose:
                print('Playing random action %s from %s' % (ret_move, fallback_move_set))
        else:
            raise Exception('No valid moves remaining: %s' % game.stringRepresentation(board))
        return ret_move


class AlphaPlayer(Player):
    def __init__(self, game, nnet, MCTS, args, verbose=False):
        super(AlphaPlayer, self).__init__(game, args, verbose)
        self.nnet = nnet
        self.MCTS = MCTS
        if self.MCTS is not None:
            self.mcts = self.MCTS(self.game, self.nnet, self.args)

    def play(self, board, verbose=None):
        verbose = super(AlphaPlayer, self).get_verbose(verbose)
        if self.MCTS is not None: # Use MCTS.
            a = np.argmax(self.mcts.getActionProb(board, temp=0))
        else: # Use vanilla network without MCTS.
            valids = self.game.getValidMoves(board, 1)
            pi, v = self.nnet.predict(board)
            pi *= valids
            sum_pi = np.sum(pi)
            if sum_pi > 0:
                pi /= sum_pi # Renormalize.
            else: # All moves all equally likely.
                pi += valids
                pi /= np.sum(pi)
                #a = np.random.choice(len(pi), p=pi) # Get random action.
                #return a
            a = np.argmax(pi)
        if verbose:
            print(f'Choosing action {a}')
        return a


class RawAlphaPlayer(AlphaPlayer):
    def __init__(self, game, nnet, args=None, verbose=False):
        super(RawAlphaPlayer, self).__init__(game, nnet, None, args, verbose)


class MCTSPlayer(AlphaPlayer):
    def __init__(self, game, MCTS, args, verbose=False):
        super(MCTSPlayer, self).__init__(game, None, MCTS, args, verbose)


class AlphaBetaPlayer(Player):
    def __init__(self, game, ABS, args, verbose=False):
        super(AlphaBetaPlayer, self).__init__(game, args, verbose)
        self.game = game
        self.ABS = ABS
        self.abs = self.ABS(self.game, self.args)

    def play(self, board, verbose=None):
        verbose = super(AlphaBetaPlayer, self).get_verbose(verbose)
        a = np.argmax(self.abs.getActionProb(board))
        if verbose:
            print(f'Choosing action {a}')        
        return a

