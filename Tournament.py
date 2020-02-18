import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time
from itertools import combinations
from Arena import Arena
from utils import *
import random


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
        self.players = players.copy()
        self.ratings = None
        self.game = game
        self.display = display


    def compete(self, num, rounds=1, rated=False, reset_ratings=False, verbose=False):
        '''
        Hosts a tournament between all players. All Participants are shuffled
        in each call.

        Args:
            num: Number of games played per pair.
            rounds: Number of rounds per tournament.
            rated: Whether to calculate elo (default False).
            reset_ratings: Whether to reset the ratings (default False).
            verbose: Whether to print progress (default False).
        
        Returns:
            A List with (wins, losses, draws) for each player if rated=False.
            A numpy array with ratings for each player if rated=True.
        '''
        eps_time = AverageMeter()
        end = time.time()
        eps = 0
        maxeps = 0

        p_idx = np.arange(0, len(self.players))
        arena_pairs = list(combinations(p_idx, 2))
        maxeps = len(arena_pairs) * rounds
        if verbose == 1:
            bar = Bar('Tournament'.ljust(10, ' '), max=maxeps)

        if reset_ratings or self.ratings is None:
            self.ratings = np.full(len(self.players), self.start_rating())

        results = []
        for p in (self.players):
            results.append({'win': 0, 'loss': 0, 'draw': 0})

        for r in range(rounds):
            random.shuffle(arena_pairs)
            for p1_idx, p2_idx in arena_pairs:
                p1 = self.players[p1_idx]
                p2 = self.players[p2_idx]
                if verbose >= 2:
                    print(f'Pitting {p1.name} vs {p2.name}.')
                arena = Arena(p1, p2, self.game, self.display)
                p1_win, p2_win, draw, ss = arena.playGames(num, return_s=True, verbose=verbose-1 if verbose>0 else 0)

                # Update metrics of player 1.
                results[p1_idx]['win'] += p1_win
                results[p1_idx]['loss'] += p2_win
                results[p1_idx]['draw'] += draw

                # Update metrics of player 2.
                results[p2_idx]['win'] += p2_win
                results[p2_idx]['loss'] += p1_win
                results[p2_idx]['draw'] += draw

                if rated:
                    for s in ss:
                        rating_p1 = self.ratings[p1_idx]
                        rating_p2 = self.ratings[p2_idx]
                        self.ratings[p1_idx] = self.new_rating(rating_p1, rating_p2, s)
                        self.ratings[p2_idx] = self.new_rating(rating_p2, rating_p1, -(s-1))

                eps += 1
                eps_time.update(time.time() - end)
                end = time.time()
                if verbose == 1:
                    bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.2f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps, maxeps=maxeps, et=eps_time.avg,
                                                                                                        total=bar.elapsed_td, eta=bar.eta_td)
                    bar.next()

        if verbose == 1:
            bar.finish()

        results = [(result['win'], result['loss'], result['draw']) for result in results]
        if rated:
            return results, np.array([rating for rating in self.ratings])
        else:
            return results, None


    def start_rating(self, r=1500):
        '''
        Returns:
            The default rating of a new player (default 1500).
        '''
        return r


    def new_rating(self, eloA, eloB, s, k=60):
        '''
        Args:
            eloA: Rating of player A.
            eloB: Rating of player B.
            s: Game result (win=1, draw=0.5, loss=0).
            k: maximum rating change between two evenly matched players (default 60).
        
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

