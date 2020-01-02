import numpy as np

"""
Random and Human-ineracting players for the game of TicTacToe.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on the OthelloPlayers by Surag Nair.

"""
class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        valid_indices = [i for i, valid in enumerate(valids) if valid]
        a = np.random.choice(valid_indices)
        return a


# TODO: Make code generic for any game.
class HumanPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                print(int(i/self.game.n), int(i%self.game.n))
        while True:
            # Python 3.x
            a = input()
            # Python 2.x 
            # a = raw_input()

            x,y = [int(x) for x in a.split(' ')]
            a = self.game.n * x + y if x!= -1 else self.game.n ** 2
            if valid[a]:
                break
            else:
                print('Invalid')

        return a


class OneStepLookaheadPlayer():
    """Simple player who always takes a win if presented, or blocks a loss if obvious, otherwise is random."""
    def __init__(self, game, verbose=True):
        self.game = game
        self.player_num = 1
        self.verbose = verbose

    def play(self, board):
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
            if self.verbose: print('Playing winning action %s from %s' % (ret_move, win_move_set))
        elif len(stop_loss_move_set) > 0:
            ret_move = np.random.choice(list(stop_loss_move_set))
            if self.verbose: print('Playing loss stopping action %s from %s' % (ret_move, stop_loss_move_set))
        elif len(fallback_move_set) > 0:
            ret_move = np.random.choice(list(fallback_move_set))
            if self.verbose: print('Playing random action %s from %s' % (ret_move, fallback_move_set))
        else:
            raise Exception('No valid moves remaining: %s' % game.stringRepresentation(board))

        return ret_move


class AlphaPlayer():
    def __init__(self, game, nnet, MCTS=None, nnargs=None):
        self.game = game
        self.nnet = nnet
        self.MCTS = MCTS
        self.nnargs = nnargs
        if self.MCTS is not None:
            self.mcts = self.MCTS(self.game, self.nnet, self.nnargs)

    def play(self, board):
        if self.MCTS is not None: # Use MCTS.
            a = np.argmax(self.mcts.getActionProb(board, temp=0))
        else: # Use vanilla network without MCTS.
            valids = self.game.getValidMoves(board, 1)
            pi, v = self.nnet.predict(board)
            pi *= valids
            sum_pi = np.sum(pi)
            if sum_pi > 0:
                pi /= sum_pi    # Renormalize.
            else: # All moves all equally likely.
                pi += valids
                pi /= np.sum(pi)
                #action = np.random.choice(len(pi), p=pi) # Get random action.
                #return action
            a = np.argmax(pi)
        return a