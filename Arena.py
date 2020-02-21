import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time

class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """
    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        curPlayer = 1
        curPlayerResigned = False
        board = self.game.getInitBoard()
        it = 0
        # Custom Input.
        boardHistory = []
        customInput = None
        while self.game.getGameEnded(board, curPlayer) == 0:
            it+=1
            if verbose:
                assert(self.display)
                self.display(board)
                print('------------------------------')
                print(f"Turn {it}: {players[curPlayer+1].name}'s turn (Player {curPlayer}).")
            player = players[curPlayer+1]
            cannonicalBoard = self.game.getCanonicalForm(board, curPlayer)
            if self.game.args.useCustomInput:
                boardHistory, customInput = self.game.getCustomInput(cannonicalBoard,
                                                                     curPlayer,
                                                                     boardHistory,
                                                                     customInput)
            action, expected_outcome, resign = player.play(cannonicalBoard,
                                                           curPlayer,
                                                           verbose=verbose>1,
                                                           customInputData=(boardHistory, customInput))
            if resign: # The player resigned.
                curPlayerResigned = True
                break
            valids = self.game.getValidMoves(cannonicalBoard, 1)

            if valids[action]==0:
                print(action)
                assert valids[action] > 0
            board, curPlayer = self.game.getNextState(board, curPlayer, action)

        if curPlayerResigned:
            result = -curPlayer
        else:
            result = curPlayer * self.game.getGameEnded(board, curPlayer)
        if verbose:
            assert(self.display)
            self.display(board)
            if int(result) != 0: # A player won, the other lost.
                print(f'Game Over: {players[result+1].name} won on turn {it} (Result: {result}).')
            else:
                print(f'Game drawn on turn {it} (Result: {result}).')
        return result

    def playGames(self, num, return_s=False, switch_players=True, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
            s: list containing game outcomes if return_s is True.
        """
        if return_s:
            ss = []
        
        eps_time = AverageMeter()
        if verbose:
            bar = Bar('Arena'.ljust(10, ' '), max=num)
        end = time.time()
        eps = 0
        maxeps = int(num//2*2)

        num = int(num/2) # Uneven numbers become the next smaller even number divided by 2.
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in range(num):
            gameResult = self.playGame(verbose=verbose-1 if verbose > 0 else 0)
            if gameResult==1:
                oneWon+=1
            elif gameResult==-1:
                twoWon+=1
            else:
                draws+=1
            
            if return_s:
                ss.append((int(gameResult)+1)/2)
            
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            if verbose == 1:
                bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.2f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps, maxeps=maxeps, et=eps_time.avg,
                                                                                                       total=bar.elapsed_td, eta=bar.eta_td)
                bar.next()

        if switch_players:
            self.player1, self.player2 = self.player2, self.player1
        else:
            oneWon, twoWon = twoWon, oneWon
        
        for _ in range(num):
            gameResult = self.playGame(verbose=verbose-1 if verbose > 0 else 0)
            if gameResult==-1:
                oneWon+=1                
            elif gameResult==1:
                twoWon+=1
            else:
                draws+=1
            
            if return_s:
                ss.append(-((int(gameResult)+1)/2-1))

            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            if verbose == 1:
                bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.2f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps, maxeps=maxeps, et=eps_time.avg,
                                                                                                       total=bar.elapsed_td, eta=bar.eta_td)
                bar.next()

        if verbose == 1:
            bar.finish()

        if switch_players:
            self.player1, self.player2 = self.player2, self.player1
        else:
            oneWon, twoWon = twoWon, oneWon
        
        if return_s:
            return oneWon, twoWon, draws, ss
        else:
            return oneWon, twoWon, draws
