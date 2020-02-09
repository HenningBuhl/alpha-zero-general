from collections import deque
from Arena import Arena
from Tournament import Tournament
from Players import AlphaPlayer
from MCTS import MCTS
import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle
from collections import deque


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        if self.args.pastOpponents is not None:
            self.opponents = []
            self.results = [] # Contains list of list of tuples (win, loss, draw) against each past opponent.
            self.ratings = []
        else:
            self.pnet = self.nnet.__class__(self.game, self.args)  # The competitor network.
            self.results = [] # Contains list of tuples (win, loss, draw) for each battle in the arena against previous version.
        self.mcts = MCTS(self.game, self.nnet, self.args)
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
            canonicalBoard = self.game.getCanonicalForm(board,self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            if self.args.useSymmetry:
                sym = self.game.getSymmetries(canonicalBoard, pi)
                for b, p in sym:
                    trainExamples.append([b, self.curPlayer, p, None])
            else:
                trainExamples.append([canonicalBoard, self.curPlayer, pi, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)

            if r!=0:
                return [(board, pi, int(r)*((-1)**(player!=self.curPlayer))) for (board, player, pi, v) in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters+1):
            # bookkeeping
            print('------ITER ' + str(i) + '------')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i>1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
    
                eps_time = AverageMeter()
                bar = Bar('Self Play', max=self.args.numEps)
                end = time.time()
    
                for eps in range(self.args.numEps):
                    self.mcts = MCTS(self.game, self.nnet, self.args)   # reset search tree
                    iterationTrainExamples += self.executeEpisode()
    
                    # bookkeeping + plot progress
                    eps_time.update(time.time() - end)
                    end = time.time()
                    bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=self.args.numEps, et=eps_time.avg,
                                                                                                               total=bar.elapsed_td, eta=bar.eta_td)
                    bar.next()
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
            shuffle(trainExamples)

            # Training new network, keeping a copy of the old one.
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.nnet.train(trainExamples)
            if self.args.pastOpponents is not None and self.args.arenaCompare is not None: # Tournament.
                print('HOSTING TOURNAMENT AGAINST PAST VERSIONS')
                #if len(self.opponents) < self.args.pastOpponents: # This line adds only one player per iteration.
                while len(self.opponents) < self.args.pastOpponents: # Only add new opponents if less than pastOpponents exist.
                    self.pnet = self.nnet.__class__(self.game, self.args)
                    self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')                
                    self.opponents.append(self.pnet)

                participant_models = [self.nnet]+list(self.opponents)
                players = [AlphaPlayer(self.game, nnet, MCTS, self.args).play for nnet in participant_models]
                tournament = Tournament(players, self.game, display=self.game.display)
                results, ratings = tournament.compete(self.args.arenaCompare, rated=True, verbose=1)
                self.results.append(results)
                self.ratings.append(ratings)
                maxEloArg = np.argmax(ratings)
                maxElo = ratings[maxEloArg]
                print(f'BEST PLAYER IS ID {maxEloArg} (ELO {maxElo})')
                print(f'\tRESULTS: {results}')
                print(f'\tRATINGS: {ratings}')
                
                opponent_ratings = ratings[1::] # Only regard the opponents' elo, not the nnet's elo.
                maxOpponentEloArg = np.argmax(opponent_ratings)
                maxOpponentElo = opponent_ratings[maxOpponentEloArg]
                if self.args.minEloImprovement is not None and maxOpponentElo * (1+self.args.minEloImprovement) > ratings[0]:
                    print('REJECTING NEW MODEL')
                    self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
                else:
                    print('ACCEPTING NEW MODEL') # Tournament result.
                    self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                    self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')  

                    if len(self.opponents) == self.args.pastOpponents: # Only remove worst opponent if new model is accepted.
                        minEloArg = np.argmin(opponent_ratings)
                        #print(f'REMOVING PLAYER WITH ID {minEloArg+1} (ELO {opponent_ratings[minEloArg]})') # Here +1, because nnet is not in this list.
                        self.opponents.pop(minEloArg)

            elif self.args.arenaCompare is not None: # Arena.
                print('PITTING AGAINST PREVIOUS VERSION')
                self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')                
                arena = Arena(AlphaPlayer(self.game, self.pnet, MCTS, self.args).play,
                            AlphaPlayer(self.game, self.nnet, MCTS, self.args).play,
                            self.game)
                results = arena.playGames(self.args.arenaCompare)
                self.results.append(results)
                pwins, nwins, draws = results
                print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))

                if self.args.updateThreshold is not None and (pwins+nwins == 0 or float(nwins)/(pwins+nwins) <= self.args.updateThreshold):
                    print('REJECTING NEW MODEL')
                    self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
                else:
                    print('ACCEPTING NEW MODEL')
                    self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                    self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')                

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
