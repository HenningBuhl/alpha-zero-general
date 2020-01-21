import numpy as np
import time

class ABS():

    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.memory = {}
	
    def getActionProb(self, canonicalBoard):
        self.start = time.time()
        _, a = self.minimax(canonicalBoard)
        probs = np.zeros(self.game.getActionSize())
        probs[a] = 1
        return probs

    def minimax(self, board, depth=0, currentPlayer=1, alpha=float('-inf'), beta=float('inf')):
        elapsed = time.time() - self.start
        if self.args.maxTime is not None and elapsed > self.args.maxTime:
            return 0, None

        if self.args.maxDepth is not None and depth > self.args.maxDepth:
            return 0, None

        gameEnded = self.game.getGameEnded(board, 1)
        if gameEnded:
            gameEnded = int(gameEnded)
            return gameEnded, None
        
        # Maximizing player.
        if currentPlayer == 1:
            bestMove = None
            maxEval = float('-inf')
            for a, valid in enumerate(self.game.getValidMoves(board, currentPlayer)):
                if valid:
                    if self.args.remember:
                        s = self.game.stringRepresentation(board)
                        if (s, a) in self.memory:
                            nextBoard, nextPlayer, score = self.memory[(s, a)]
                        else:
                            nextBoard, nextPlayer = self.game.getNextState(board, currentPlayer, a)
                            score, _ = self.minimax(nextBoard, depth + 1, nextPlayer, alpha, beta)
                            self.memory[(s, a)] = (nextBoard, nextPlayer, score)
                    else:
                        nextBoard, nextPlayer = self.game.getNextState(board, currentPlayer, a)
                        score, _ = self.minimax(nextBoard, depth + 1, nextPlayer, alpha, beta)

                    if score > maxEval:
                        bestMove = a
                        maxEval = score

                    if self.args.prune:
                        alpha = np.maximum(alpha, score)
                        if beta <= alpha: # Prune.
                            break
            return maxEval, bestMove

        # Minimizing Player.
        else: # player == -1
            bestMove = None
            minEval = float('inf')
            for a, valid in enumerate(self.game.getValidMoves(board, currentPlayer)):
                if valid:
                    nextBoard, nextPlayer = self.game.getNextState(board, currentPlayer, a)
                    score, _ = self.minimax(nextBoard, depth + 1, nextPlayer, alpha, beta)
                    if score < minEval:
                        bestMove = a
                        minEval = score
                    
                    if self.args.prune:
                        beta = np.minimum(beta, score)
                        if beta <= alpha: # Prune.
                            break
            return minEval, bestMove

