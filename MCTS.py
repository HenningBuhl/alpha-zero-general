import math
import numpy as np
import time
import itertools

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)
        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        
        start = time.time()
        for i in range(self.args.numMCTSSims) if self.args.numMCTSSims is not None else itertools.count():
            self.search(canonicalBoard, True)
            elapsed = time.time() - start
            if self.args.maxTime is not None and elapsed > self.args.maxTime:
                break
        
        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp==0:
            bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA] = 1
        else:
            counts = [x**(1./temp) for x in counts]
            counts_sum = float(sum(counts))
            probs = [x/counts_sum for x in counts]
        return probs

    def search(self, canonicalBoard, rootNode=False):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.game.stringRepresentation(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s]!=0:
            # Terminal node.
            return -int(self.Es[s])

        if s not in self.Ps:
            # Leaf node.
            valids = self.game.getValidMoves(canonicalBoard, 1)

            if self.args.rollout == 'random':
                self.Ps[s] = valids / np.sum(valids)
                board = canonicalBoard
                cur_player = 1
                while True: # Random rollout.
                    vs = self.game.getValidMoves(board, cur_player)
                    a = np.random.choice(vs.shape[0], p=vs/np.sum(vs))
                    board, next_player = self.game.getNextState(board, cur_player, a)
                    r = self.game.getGameEnded(board, 1)
                    if r != 0:
                        break
                    cur_player = next_player
                v = r
            elif self.args.rollout == 'single':
                self.Ps[s], v = self.nnet.predict(canonicalBoard)
                self.Ps[s] = self.Ps[s]*valids # masking invalid moves
                sum_Ps_s = np.sum(self.Ps[s])
                if sum_Ps_s > 0:
                    self.Ps[s] /= sum_Ps_s # renormalize
                else:
                    # if all valid moves were masked make all valid moves equally probable
                    # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                    # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                    print("All valid moves were masked, do workaround.")
                    self.Ps[s] = self.Ps[s] + valids
                    self.Ps[s] /= np.sum(self.Ps[s])
            elif self.args.rollout == 'fast':
                self.Ps[s] = valids / np.sum(valids)
                board = canonicalBoard
                cur_player = 1
                while True: # Fast rollout.
                    vs = self.game.getValidMoves(board, cur_player)
                    pi_fast = self.nnet.predict_fast(board)
                    pi_fast = pi_fast * vs
                    a = np.random.choice(vs.shape[0], p=pi_fast/np.sum(pi_fast))
                    board, next_player = self.game.getNextState(board, cur_player, a)
                    r = self.game.getGameEnded(board, 1)
                    if r != 0:
                        break
                    cur_player = next_player
                v = r
            elif self.args.rollout == 'slow':
                self.Ps[s] = valids / np.sum(valids)
                board = canonicalBoard
                cur_player = 1
                while True: # Slow rollout.
                    vs = self.game.getValidMoves(board, cur_player)
                    pi, _ = self.nnet.predict(board)
                    pi = pi * vs
                    a = np.random.choice(vs.shape[0], p=pi/np.sum(pi))
                    board, next_player = self.game.getNextState(board, cur_player, a)
                    r = self.game.getGameEnded(board, 1)
                    if r != 0:
                        break
                    cur_player = next_player
                v = r            
            else:
                raise ValueError(f'rollout \'{self.args.rollout}\' is not supported.')

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -int(v)

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1
        
        # Dirichlet Noise.
        useDirNoise = False
        dirEpsilon = self.args.dirEpsilon
        if rootNode and dirEpsilon > 0:
            useDirNoise = True
            dirAlpha = self.args.dirAlpha
            dirEta = np.random.dirichlet([dirAlpha] * len(valids))
        
        # pick the action with the highest upper confidence bound
        for i, a in enumerate(range(self.game.getActionSize())):
            if valids[a]:
                if useDirNoise:
                    p = (1 - dirEpsilon) * self.Ps[s][a] + dirEpsilon * dirEta[i]
                else:
                    p = self.Ps[s][a]
                
                if (s,a) in self.Qsa:
                    u = self.Qsa[(s,a)] + self.args.cpuct*p*math.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)])
                else:
                    u = self.args.cpuct*p*math.sqrt(self.Ns[s])

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s)

        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1
        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1
        return -int(v)

