import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time
from itertools import combinations
from Arena import Arena
from utils import *

class Tournament():
    '''
    A Tournament class for staging a round robin league.
    '''
    def __init__(self, players, game, display=None):
        '''
        Args:
            players: A list of players.
            game: The game in which the player compete.
            display: A function that takes a board as input as renders it.
        
        Returns:
            An instance of the class.
        '''
        self.players = players
        self.game = game
        self.display = display


    def compete(self, num, rated=False, ratings=None, verbose=False):
        '''
        Args:
            num: Number of games player per pair.
            rated: Whether to calculate elo (default False).
            ratings: List of ratings of all players (default None).
            verbose: Whether to print progress (default False).
        
        Returns:
            A List with (wins, losses, draws) for each player if rated=False.
            A List with ratings for each player if rated=True.
        '''
        eps_time = AverageMeter()
        end = time.time()
        eps = 0
        maxeps = 0
        for i in combinations(self.players, 2): maxeps += 1
        if verbose == 1:
            bar = Bar('Tournament.compete', max=maxeps)

        if rated and ratings is None:
            ratings = [self.start_rating()] * len(self.players)

        results = {}
        for p in (self.players):
            results[p] = {'win': 0, 'loss': 0, 'draw': 0}

        for p1, p2 in combinations(self.players, 2):
            if verbose >= 2:
                print(f'Pitting {p1} vs {p2}.')
            arena = Arena(p1, p2, self.game, self.display)
            p1_win, p2_win, draw, ss = arena.playGames(num, return_s=True, verbose=verbose-1 if verbose>0 else 0)

            # List indices.
            p1_idx = self.players.index(p1)
            p2_idx = self.players.index(p2)

            # Update metrics of player 1.
            results[p1]['win'] += p1_win
            results[p1]['loss'] += p2_win
            results[p1]['draw'] += draw

            # Update metrics of player 2.
            results[p2]['win'] += p2_win
            results[p2]['loss'] += p1_win
            results[p2]['draw'] += draw

            if rated:
                for s in ss:
                    rating_p1 = ratings[p1_idx]
                    rating_p2 = ratings[p2_idx]
                    ratings[p1_idx] = self.new_rating(rating_p1, rating_p2, s)
                    ratings[p2_idx] = self.new_rating(rating_p2, rating_p1, -(s-1))

            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            if verbose == 1:
                bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps, maxeps=maxeps, et=eps_time.avg,
                                                                                                       total=bar.elapsed_td, eta=bar.eta_td)
                bar.next()

        if verbose == 1:
            bar.finish()

        results = [(p['win'], p['loss'], p['draw']) for p in results.values()]
        if rated:
            return results, ratings
        else:
            return results, None


    def start_rating(self, r=1500):
        '''
        Returns:
            The default rating of a new player (default 1000).
        '''
        return r


    def new_rating(self, eloA, eloB, s, k=60):
        '''
        Args:
            eloA: Rating of player A.
            eloB: Rating of player B.
            s: Game result (win=1, draw=0.5, loss=0).
            k: maximum rating change between two evenly matched players (default 32).
        
        Returns:
            The new rating of player A.
        '''
        return eloA + k * (s - self.expected_victory(eloA, eloB))


    def expected_victory(self, eloA, eloB):
        '''
        Args:
            eloA: Rating of player A.
            eloB: Rating of player B.
        
        Returns:
            The expected victory outcome of player A.
        '''
        return 1 / (1 + 10**((eloB - eloA) / 400))

