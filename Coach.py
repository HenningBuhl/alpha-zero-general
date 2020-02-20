from collections import deque
from Arena import Arena
from Tournament import Tournament
from Players import AlphaPlayer
from MCTS import MCTS
import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time, os, sys
from pickle import Pickler, Unpickler
from collections import deque
from torch import multiprocessing as mp


class Coach():
    '''
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    '''
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        if self.args.tournamentCompare is not None:
            self.opponents = []
            self.results = [] # Contains list of list of tuples (win, loss, draw) against each past opponent.
            self.ratings = [] # Contains a list of tuples with elo ratings against each past opponent.
        else:
            self.pnet = self.nnet.__class__(self.game, self.args)  # The competitor network.
            self.results = [] # Contains list of tuples (win, loss, draw) for each battle in the arena against previous version.
        self.player = AlphaPlayer(self.game, self.nnet, MCTS, self.args)
        self.trainExamplesHistory = [] # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False # can be overriden in loadTrainExamples()


    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            _, _, resign, pi = self.player.play(canonicalBoard, self.curPlayer, return_pi=True, temp=temp)
            if self.args.useSymmetry:
                sym = self.game.getSymmetries(canonicalBoard, pi)
                if self.args.uniqueSymmetry: # Removes duplicate symmetries.
                    #print(f'SYMMETRIES BEFORE DUPLICATE REMOVAL: {len(sym)}')
                    #for b, p in sym: self.game.display(b)
                    unique_sym = []
                    for b, p in sym:
                        if np.array([np.array_equal(p, _p) for (_, _p) in unique_sym]).sum() == 0: # If not yet in unique_sym.
                            unique_sym.append((b, p))
                    sym = unique_sym
                    #print(f'SYMMETRIES AFTER DUPLICATE REMOVAL: {len(sym)}')
                for b, p in sym:
                    trainExamples.append([b, self.curPlayer, p, None])
            else:
                trainExamples.append([canonicalBoard, self.curPlayer, pi, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)
            
            if self.args.resignThreshold is not None and resign:
                r = -self.curPlayer
            else:
                r = self.game.getGameEnded(board, self.curPlayer)

            if r != 0:
                return [(board, pi, int(r)*((-1)**(player!=self.curPlayer))) for (board, player, pi, v) in trainExamples]


    def learn(self, verbose=2):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters+1):
            # bookkeeping
            print(f'------ Iteration {i} ------')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i>1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                eps_time = AverageMeter()
                if verbose:
                    bar = Bar('Self Play'.ljust(10, ' '), max=self.args.numEps)
                end = time.time()

                for eps in range(self.args.numEps):
                    self.player.reset() # Reset the monte carlo search tree.
                    iterationTrainExamples += self.executeEpisode() # PARALLELIZE

                    # bookkeeping + plot progress
                    eps_time.update(time.time() - end)
                    end = time.time()
                    bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.2f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=self.args.numEps, et=eps_time.avg,
                                                                                                               total=bar.elapsed_td, eta=bar.eta_td)
                    if verbose:
                        bar.next()
                if verbose:
                    bar.finish()

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)
                
            if self.args.numItersForTrainExamplesHistory is not None and len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                #print("len(trainExamplesHistory) =", len(self.trainExamplesHistory), " => remove the oldest trainExamples")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i-1)
            
            # Shuffle examples before training.
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)

            # Training new network, keeping a copy of the old one.
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.nnet.train(trainExamples, verbose=verbose>0) # PARALLELIZE
            accept_model = True # Model has to be rejected explicitly.
            if self.args.tournamentCompare is not None: # Tournament.
                if verbose>1:
                    print('HOSTING TOURNAMENT AGAINST PAST VERSIONS')
                #if len(self.opponents) < self.args.pastOpponents: # This line adds only one player per iteration.
                while len(self.opponents) < self.args.pastOpponents: # Add new opponents while less than pastOpponents exist (easier to plot, but longer).
                    self.pnet = self.nnet.__class__(self.game, self.args)
                    self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')                
                    self.opponents.append(self.pnet)

                participant_models = [self.nnet] + self.opponents
                players = [AlphaPlayer(self.game, nnet, MCTS, self.args) for nnet in participant_models]
                tournament = Tournament(players, self.game, display=self.game.display)
                results, ratings = tournament.compete(self.args.tournamentCompare, # PARALLELIZE
                                                      rounds=self.args.tournamentRounds,
                                                      rated=True, verbose=verbose>0)
                self.results.append(results)
                self.ratings.append(ratings)
                maxEloArg = np.argmax(ratings)
                maxElo = ratings[maxEloArg]
                if verbose>1:
                    print(f'\tBEST PLAYER IS ID {maxEloArg} (ELO {maxElo})')
                if verbose:
                    print(f'\tResults: {results}')
                    print(f'\tRatings: {ratings}')
                
                opponent_ratings = ratings[1::] # Only regard the opponents' elo, not the nnet's elo.
                maxOpponentEloArg = np.argmax(opponent_ratings)
                maxOpponentElo = opponent_ratings[maxOpponentEloArg]
                if self.args.minEloImprovement is not None and maxOpponentElo * (1+self.args.minEloImprovement) > ratings[0]:
                    accept_model = False
                else:
                    if len(self.opponents) == self.args.pastOpponents: # Only remove worst opponent if new model is accepted.
                        minEloArg = np.argmin(opponent_ratings)
                        if verbose>1:
                            print(f'REMOVING PLAYER WITH ID {minEloArg+1} (ELO {opponent_ratings[minEloArg]})') # Here +1, because nnet is not in this list.
                        self.opponents.pop(minEloArg)

            elif self.args.arenaCompare is not None: # Arena.
                if verbose>1:
                    print('PITTING AGAINST PREVIOUS VERSION')
                self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')                
                arena = Arena(AlphaPlayer(self.game, self.nnet, MCTS, self.args),
                                AlphaPlayer(self.game, self.pnet, MCTS, self.args),
                                self.game)
                result = arena.playGames(self.args.arenaCompare, verbose=verbose>0) # PARALLELIZE
                self.results.append(result)
                nwins, pwins, draws = result
                wins = nwins+pwins
                if verbose:
                    print(f'\tResult: {result}', end='')
                    if wins > 0: # nwins+pwins is not 0. Can be used in denominator.
                        print(f' - Winrate: {nwins/(wins):.2f}')
                    else:
                        print('') # Finish line.

                if self.args.updateThreshold is not None and (wins == 0 or nwins/(wins) < self.args.updateThreshold):
                    accept_model = False

            if accept_model:
                if verbose:
                    print('Accepting new model')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
            else: # Rejected.
                if verbose:
                    print('Rejecting new model')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')


    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'


    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration)+".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed


    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile+".examples"
        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            f.closed
            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True

