import numpy as np
from utils import *


class Player():
    '''
    This class represents a Player competing in a game.
    '''
    def __init__(self, game, args=None, name=None, verbose=False):
        '''
        Args:
            game: The game to play.
            args: The arguments (default None).
            name: The name of the player (default None). Is set to class name if None.
            verbose: The verbosity (default False)
        
        Returns:
            A player object.
        '''
        self.game = game
        self.args = args
        self.verbose = verbose
        self.name = name
        if self.name is None:
            self.name = type(self).__name__ # Default name equals class name.

    def reset(self):
        '''
        Reset any data that is created/expanded while playing.
        '''
        pass

    def play(self, board, curPlayer, verbose=False, customInputData=None):
        '''
        Args:
            board: The cannoncal board.
            curPlayer: The current player.
            verbose: Whether to output information about the decision process (default False).
            customInputData: custom board view used by Neural Network when args.useCustomInput is True (default None).
        
        Returns:
            A tuple of (best_action, expected_outcome, resign).
        '''
        pass

    def get_verbose(self, verbose=None):
        '''
        Args:
            verbose: (default None).
        
        Returns:
            The verbosity level.
        '''
        return self.verbose if verbose is None else verbose

    def talk(self, a, v, r, verbose=True):
        '''
        Prints various information regarding the decision process of the player.

        Args:
            a: The action performed.
            v: The expected outcome of the game.
            r: Whether the player resigned the game.
            verbose: Whether to print or not (default True).
        '''
        if verbose:
            if v is not None:
                # Print expected outcome.
                print(f'{self.name}\'s outcome prediction: {v} (absolute view)')
            if r:
                # Print whether the player resigned the game.
                print(f'{self.name} resigned the game.')
            else:
                # Print which action the player chose.
                print(f'{self.name} chose action {a}')


class RandomPlayer(Player):
    def __init__(self, game, args=None, name=None, verbose=False):
        super(RandomPlayer, self).__init__(game, args=args, name=name, verbose=verbose)

    def play(self, board, curPlayer, verbose=None, customInputData=None):
        verbose = self.get_verbose(verbose=verbose)
        valids = self.game.getValidMoves(board, 1)
        valid_indices = [i for i, valid in enumerate(valids) if valid]
        a = np.random.choice(valid_indices) # Chose random action.
        self.talk(a, None, False, verbose=verbose)
        return a, None, False


class HumanPlayer(Player):
    def __init__(self, game, args=None, name=None, verbose=False):
        super(HumanPlayer, self).__init__(game, args=args, name=name, verbose=verbose)

    def play(self, board, curPlayer, verbose=None, customInputData=None):
        verbose = self.get_verbose(verbose=verbose)
        valid_moves = self.game.getValidMoves(board, 1)
        uf_moves = self.game.getUserFriendlyMoves(board, 1)
        print('Pass -1 into action prompt to resign.')
        for i, (valid, uf_move) in enumerate(uf_moves): # Display available actions.
            if valid:
                if self.verbose:
                    print(f'Move: {i:2d}, UF: {uf_move}')

        while True: # Don't continue until user input is fetched.
            a = int(input())
            if a == -1: # HumanPlayers wants to resign
                break
            else: # HumanPlayer wants to play an action.
                if valid_moves[move]:
                    break
                else:
                    if verbose:
                        print('Invalid move')

        resign = a == -1 # HumanPlayer can always resign.
        self.talk(a, None, resign, verbose=verbose)
        return a, None, resign


class GreedyPlayer(Player):
    """Simple player who always takes a win if presented, or blocks a loss if obvious, otherwise is random."""
    def __init__(self, game, args=None, name=None, verbose=False):
        super(GreedyPlayer, self).__init__(game, args=args, name=name, verbose=verbose)
        self.player_num = 1

    def play(self, board, curPlayer, verbose=None, customInputData=None):
        verbose = self.get_verbose(verbose=verbose)
        valid_moves = self.game.getValidMoves(board, self.player_num)
        win_move_set = set() # Actions that result in a win.
        stop_loss_move_set = set() # Actions that block a loss.
        fallback_move_set = set() # All other actions (random).
        
        for move, valid in enumerate(valid_moves):
            if not valid: continue
            if -self.player_num == self.game.getGameEnded(*self.game.getNextState(board, self.player_num, move)):
                win_move_set.add(move)
            elif -self.player_num == self.game.getGameEnded(*self.game.getNextState(board, -self.player_num, move)):
                stop_loss_move_set.add(move)
            else:
                fallback_move_set.add(move)

        if len(win_move_set) > 0:
            a = np.random.choice(list(win_move_set))
        elif len(stop_loss_move_set) > 0:
            a = np.random.choice(list(stop_loss_move_set))
        elif len(fallback_move_set) > 0:
            a = np.random.choice(list(fallback_move_set))
        else:
            raise Exception(f'No valid moves remaining: {game.stringRepresentation(board)}')
        
        self.talk(a, None, False, verbose=verbose)
        return a, None, False


class AlphaPlayer(Player):
    def __init__(self, game, nnet, MCTS, args, name=None, verbose=False):
        super(AlphaPlayer, self).__init__(game, args=args, name=name, verbose=verbose)
        self.nnet = nnet
        self.MCTS = MCTS
        self.reset()

    def reset(self):
        if self.MCTS is not None:
            self.mcts = self.MCTS(self.game, self.nnet, self.args)

    def play(self, board, curPlayer, return_pi=False, temp=0, verbose=None, customInputData=None):
        verbose = self.get_verbose(verbose=verbose)
        if self.MCTS is not None: # AlphaPlayer or MCTSPlayer.
            pi, v = self.mcts.getActionProb(board, temp=temp, customInputData=customInputData)
            a = np.argmax(pi)
                # Save time and don't calculate v is resignation is disabled?
                _, v = self.nnet.predict(board)
                v = v[0] # Convert [v] into v.
        else: # RawAlphaPlayer.
            if self.game.args.useCustomInput:
                pi, v = self.nnet.predict(customInputData[1])
            else:
                pi, v = self.nnet.predict(board)
            v = v[0] # Convert [v] into v.
            valids = self.game.getValidMoves(board, 1)
            pi *= valids
            sum_pi = np.sum(pi)
            if sum_pi > 0:
                pi /= sum_pi # Renormalize.
            else: # All moves all equally likely.
                pi += valids
                pi /= np.sum(pi)
                #a = np.random.choice(len(pi), p=pi) # Get random action. # RawNet should be deterministic.
            a = np.argmax(pi)
        
        resign = False
        if self.args.resignThreshold is not None:
            resign = v <= self.args.resignThreshold

        v *= curPlayer # Convert cannonical v into absolute v.
        self.talk(a, v, resign, verbose=verbose)
        
        if return_pi:
            return a, v, resign, pi
        else:
            return a, v, resign


class RawAlphaPlayer(AlphaPlayer):
    def __init__(self, game, nnet, args=None, name=None, verbose=False):
        super(RawAlphaPlayer, self).__init__(game, nnet, None, args=args, name=name, verbose=verbose)


class MCTSPlayer(AlphaPlayer):
    def __init__(self, game, MCTS, args, name=None, verbose=False):
        super(MCTSPlayer, self).__init__(game, None, MCTS, args=args, name=name, verbose=verbose)


class ABTSPlayer(Player):
    def __init__(self, game, ABTS, args, name=None, verbose=False):
        super(ABTSPlayer, self).__init__(game, args=args, name=name, verbose=verbose)
        self.game = game
        self.ABTS = ABTS
        self.reset()

    def reset(self):
        self.abts = self.ABTS(self.game, self.args)

    def play(self, board, curPlayer, verbose=None, customInputData=None):
        verbose = self.get_verbose(verbose=verbose)
        probs, v = self.abts.getActionProb(board)
        if np.sum(probs) == 0: # ABTS didn't have enough time/depth to find a move.
            vs = self.game.getValidMoves(board, 1)
            probs = vs / np.sum(vs) # All moves are equally likely
        a = np.argmax(probs)
        resign = False
        if self.args.resignThreshold is not None:
            resign = v <= self.args.resignThreshold
        v *= curPlayer # Convert cannonical v into absolute v.
        self.talk(a, v, resign, verbose=verbose)    
        return a, v, resign

