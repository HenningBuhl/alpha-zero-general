from collections import deque
from Arena import Arena
from Tournament import Tournament
from Players import AlphaPlayer
from MCTS import MCTS
import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time, os, sys
from pickle import Pickler, Unpickler
from torch.multiprocessing import set_start_method
from torch import multiprocessing as mp
import contextlib


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

        if self.args.parallelize:
            try:
                #set_start_method('fork', force=True)
                set_start_method('spawn', force=True)
                #set_start_method('forkserver', force=True)
            except RuntimeError:
                print('set_start_method DID NOT WORK')


    def executeEpisode(self, lock=contextlib.suppress()):
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
        # Custom Input variables.
        boardHistory = []
        customInput = None
        
        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)
            if self.game.args.useCustomInput: # Construct custom input.
                boardHistory, customInput = self.game.getCustomInput(canonicalBoard, self.curPlayer, boardHistory, customInput)
            with lock:
                _, _, resign, pi = self.player.play(canonicalBoard, self.curPlayer, return_pi=True, temp=temp, customInputData=(boardHistory, customInput))

            if self.args.useSymmetry:
                if self.game.args.useCustomInput:
                    sym = self.game.getCustomSymmetries(customInput, pi)
                else:
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
                if self.game.args.useCustomInput:
                    trainExamples.append([customInput, self.curPlayer, pi, None])
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


    def selfplayParent(self, results, messages, trainData, trainDataLock, stateValue, stateLock, checkpointLock, iterationValue, iterationLock):
        messages.put('Self play process started.')

        for i in range(1, self.args.numIters+1):
            messages.put(f'Self play iteration {i}.')
            iterData = deque([], maxlen=self.args.maxlenOfQueue)
            
            for eps in range(self.args.numEps):
                #messages.put(f'Self play episode {eps+1}')
                self.player.reset()
                iterData += self.executeEpisode(lock=checkpointLock)
            
            #messages.put(f'Self play data added to shared memmory.')
            with trainDataLock: # Add iteration data to trainData.
                trainData.append(iterData)
                if self.args.numItersForTrainExamplesHistory is not None and len(trainData) > self.args.numItersForTrainExamplesHistory:
                    #messages.put(f'Removing oldest training data.')
                    trainData.pop(0)
            
            if self.args.numSelfplayWorker == 1:
                self.saveTrainExamples(i-1)

            with stateLock:
                if stateValue.value == 0:
                    #messages.put('Unlocked state value to 1.')
                    stateValue.value = 1

        messages.put(f'Self play finished.')
        with stateLock: # Signal self play has finished.
            stateValue.value = -1


    def trainingParent(self, results, messages, trainData, trainDataLock, stateValue, stateLock, checkpointLock, iterationValue, iterationLock):
        messages.put('Training process started.')

        while 1:
            with stateLock:
                if stateValue.value == -1: # Check if self play process has terminated.
                    #messages.put('Training process finished.')
                    break

            with stateLock:
                #messages.put('Waiting for training data.') # Prints thousands of times per second.
                if stateValue.value < 1: # Check if training data exists.
                    continue
            
            with trainDataLock:
                #messages.put('Copying training data from shared memory.')
                trainDataCopy = []
                for e in trainData:
                    trainDataCopy.append(e)

            #messages.put('Training data is available.')
            trainExamples = []
            for e in trainDataCopy:
                trainExamples.extend(e)
            with checkpointLock:
                #messages.put('Saving nnet checkpoint.')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            
            # Train.
            #messages.put('Training model on data.')
            self.nnet.train(trainExamples, lock=checkpointLock, verbose=0)
            with iterationLock:
                iterationValue.value += 1

            # Print average losses.
            m = 50 # Average of last m batches
            n = np.minimum(m, len(self.nnet.history['loss'])) # Normalizer.
            avg_loss = np.sum(np.array(self.nnet.history['loss'])[-m:]) / n
            avg_pi_loss = np.sum(np.array(self.nnet.history['pi_loss'])[-m:]) / n
            avg_v_loss = np.sum(np.array(self.nnet.history['v_loss'])[-m:]) / n
            messages.put(f'Loss: {avg_loss:5.3f}, Pi Loss: {avg_pi_loss:5.3f}, V Loss: {avg_v_loss:5.3f}')

            with stateLock:
                if stateValue.value == 1:
                    #messages.put('Unlocked state value to 2.')
                    stateValue.value = 2

        # Save results.
        #messages.put('Adding loss metrics to results.')
        results['loss'].extend(self.nnet.history['loss'])
        results['pi_loss'].extend(self.nnet.history['pi_loss'])
        results['v_loss'].extend(self.nnet.history['v_loss'])


    def compareParent(self, results, messages, trainData, trainDataLock, stateValue, stateLock, checkpointLock, iterationValue, iterationLock):
        messages.put('Compare process started.')
        
        while 1:
            with stateLock:
                if stateValue.value == -1: # Check if self play process has terminated.
                    #messages.put('Compare process finished.')
                    break
            
            with stateLock:
                if stateValue.value < 2: # Check if trained model exists.
                    continue
            
            accept_model = True # Model has to be rejected explicitly.
            if self.args.tournamentCompare is not None: # Tournament.
                while len(self.opponents) < self.args.pastOpponents: # Add new opponents while less than pastOpponents exist (easier to plot, but longer).
                    self.pnet = self.nnet.__class__(self.game, self.args)
                    with checkpointLock:
                        self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')                
                    self.opponents.append(self.pnet)
                participant_models = [self.nnet] + self.opponents
                players = [AlphaPlayer(self.game, nnet, MCTS, self.args) for nnet in participant_models]
                tournament = Tournament(players, self.game)
                _results, ratings = tournament.compete(self.args.tournamentCompare,
                                                      rounds=self.args.tournamentRounds,
                                                      rated=True, verbose=0)
                results['results'].append(_results)
                results['ratings'].append(ratings)
                maxEloArg = np.argmax(ratings)
                maxElo = ratings[maxEloArg]

                msg = f'Tournament ratings {ratings}'

                opponent_ratings = ratings[1::] # Only regard the opponents' elo, not the nnet's elo.
                maxOpponentEloArg = np.argmax(opponent_ratings)
                maxOpponentElo = opponent_ratings[maxOpponentEloArg]
                if self.args.minEloImprovement is not None and maxOpponentElo * (1+self.args.minEloImprovement) > ratings[0]:
                    accept_model = False
                else:
                    if len(self.opponents) == self.args.pastOpponents: # Only remove worst opponent if new model is accepted.
                        minEloArg = np.argmin(opponent_ratings)
                        self.opponents.pop(minEloArg)

            elif self.args.arenaCompare is not None:
                with checkpointLock:
                    self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')                
                arena = Arena(AlphaPlayer(self.game, self.nnet, MCTS, self.args),
                              AlphaPlayer(self.game, self.pnet, MCTS, self.args),
                              self.game)
                result = arena.playGames(self.args.arenaCompare, verbose=0)
                results['results'].append(result)
                nwins, pwins, draws = result
                wins = nwins+pwins
                
                msg = f'Arena result: {result}'
                if wins > 0: # nwins+pwins is not 0. Can be used in denominator.
                    msg += f' - Winrate: {100*nwins/(wins):.0f}%'

                if self.args.updateThreshold is not None and (wins == 0 or nwins/(wins) < self.args.updateThreshold):
                    accept_model = False
            else:
                continue # Do not compare at all.
            
            if accept_model:
                messages.put(msg + ' - Accepting new model.')
                with checkpointLock:
                    with iterationLock:
                        self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(iterationValue.value)) # Safe, because nothing is loaded from disk.
                    self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
            else: # Rejected.
                messages.put(msg + ' - Rejecting new model.')
                with checkpointLock:
                    self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')


    def learn(self, verbose=1):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        if self.args.parallelize:
            # Prepare process arguments.
            manager = mp.Manager()
            messages = manager.Queue() # 
            results = manager.dict() # Dict with results (loss, pi_loss, v_loss).
            results['loss'] = manager.list()
            results['pi_loss'] = manager.list()
            results['v_loss'] = manager.list()
            results['results'] = manager.list()
            results['ratings'] = manager.list()
            trainData = manager.list() # List of training data (LOCK REQURIED).
            trainDataLock = manager.Lock() # Lock used for adding training examples and copying it.
            stateValue = manager.Value('d', 0) # Value indicating the state of the training process (LOCK REQURIED).
            # stateValue: 0: no data, 1: no trained model, -1: self play done.
            stateLock = manager.Lock() # Lock used for changing stateValue or executing sensitive, state specific work.
            checkpointLock = manager.Lock() # Lock used for checkpoint save/load operation.
            iterationValue = manager.Value('d', 0) # Current iteration of nnet trainig (not self play).
            iterationLock = manager.Lock() # Lock used to access the current iteration.
            
            # Process arguments.
            pargs = (results, messages,
                     trainData, trainDataLock,
                     stateValue, stateLock,
                     checkpointLock,
                     iterationValue, iterationLock)
            
            # Create processes.
            print('Create processes.')
            ps = []
            for _ in range(self.args.numSelfplayWorker): # Using more than 1 numSelfplayWorker  will result in no training data being saved to disk.
                ps.append(mp.Process(target=self.selfplayParent, args=pargs))
            for _ in range(self.args.numTrainWorker):
                ps.append(mp.Process(target=self.trainingParent, args=pargs))
            ps.append(mp.Process(target=self.compareParent, args=pargs)) # Only 1 compareWorker.

            # Start processes.
            print('Start processes.')
            for p in ps:
                p.start()

            # Print messages from parents.
            while 1:
                with stateLock:
                    if stateValue.value == -1:
                        break
                if not messages.empty():
                    msg = messages.get()
                    print(msg)

            # Join processes.
            print('Join processes.')
            for p in ps:
                p.join()
            
            # Set nnet history to multiprocessing results.
            self.nnet.history['loss'] = list(results['loss'])
            self.nnet.history['pi_loss'] = list(results['pi_loss'])
            self.nnet.history['v_loss'] = list(results['v_loss'])
            self.results = list(results['results'])
            self.ratings = list(results['ratings'])
            return
        
        for i in range(1, self.args.numIters+1):
            print(f'------ Iteration {i} ------')
            if not self.skipFirstSelfPlay or i>1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                eps_time = AverageMeter()
                if verbose:
                    bar = Bar('Self Play'.ljust(10, ' '), max=self.args.numEps)
                end = time.time()

                for eps in range(self.args.numEps):
                    self.player.reset() # Reset the monte carlo search tree.
                    iterationTrainExamples += self.executeEpisode()

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
            self.nnet.train(trainExamples, verbose=verbose>0)
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
                results, ratings = tournament.compete(self.args.tournamentCompare,
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
                result = arena.playGames(self.args.arenaCompare, verbose=verbose>0)
                self.results.append(result)
                nwins, pwins, draws = result
                wins = nwins+pwins
                if verbose:
                    print(f'\tResult: {result}', end='')
                    if wins > 0: # nwins+pwins is not 0. Can be used in denominator.
                        print(f' - Winrate: {100*nwins/(wins):.0f}%')
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

