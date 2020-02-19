import numpy as np
import time

class ABTS():
    '''
    This class handles the Alpha-Beta search tree.
    '''
    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.memory = {}
	
    def getActionProb(self, canonicalBoard):
        '''
        Args:
            canonicalBoard: The cannonical board.
            currentPlayer: The current player.

        Returns:
            Probability distribution over all possible moves (best move has value 1, all others 0).
        '''
        self.start = time.time()
        a, max_eval = self.minimax(canonicalBoard)
        probs = np.zeros(self.game.getActionSize())
        if a == -1:
            #print('COULDN\'T FIND A MOVE. CHOOSING FIRST AVAILABLE MOVE.')
            pass
        else:
            probs[a] = 1
        return probs, max_eval
    

    def minimax(self, board, depth=0, currentPlayer=1, alpha=float('-inf'), beta=float('inf')):
        '''
        Args:
            board: Current cannonical board.
            depth: The current depth in the seach tree (default 0).
            currentPlayer: The current player from the cannonical view (default 1).
            alpha: Alpha used for pruning (default -inf).
            beta: Beta used for pruning (default inf).
        
        Returns:
            A tuple (best_move, eval). If game state is terminal, best_move is None.
        '''
        elapsed = time.time() - self.start

        # Game ended.
        gameEnded = self.game.getGameEnded(board, 1)
        if gameEnded:
            return -1, gameEnded
        
        # Max time reached.
        if self.args.maxTime is not None and elapsed > self.args.maxTime:
            return -1, gameEnded

        # Max depth reached.
        if self.args.maxDepth is not None and depth > self.args.maxDepth:
            return -1, gameEnded
        
        # Maximizing player.
        if currentPlayer == 1:
            bestMove = -1
            maxEval = float('-inf')
            for a, valid in enumerate(self.game.getValidMoves(board, currentPlayer)):
                if valid:
                    if self.args.remember:
                        s = self.game.stringRepresentation(board)
                        if (s, a, currentPlayer) in self.memory:
                            nextBoard, nextPlayer, score = self.memory[(s, a, currentPlayer)]
                        else:
                            nextBoard, nextPlayer = self.game.getNextState(board, currentPlayer, a)
                            _, score = self.minimax(nextBoard, depth + 1, nextPlayer, alpha, beta)
                            self.memory[(s, a, currentPlayer)] = (nextBoard, nextPlayer, score)
                    else:
                        nextBoard, nextPlayer = self.game.getNextState(board, currentPlayer, a)
                        _, score = self.minimax(nextBoard, depth + 1, nextPlayer, alpha, beta)

                    #if depth == 0:
                    #    print(f'Action {a}, Score: {score}')

                    if score > maxEval:
                        bestMove = a
                        maxEval = score

                    if self.args.prune:
                        alpha = np.maximum(alpha, score)
                        if beta <= alpha: # Prune.
                            break
            return bestMove, maxEval

        # Minimizing Player.
        else: # currentPlayer == -1
            bestMove = -1
            minEval = float('inf')
            for a, valid in enumerate(self.game.getValidMoves(board, currentPlayer)):
                if valid:
                    if self.args.remember:
                        s = self.game.stringRepresentation(board)
                        if (s, a, currentPlayer) in self.memory:
                            nextBoard, nextPlayer, score = self.memory[(s, a, currentPlayer)]
                        else:
                            nextBoard, nextPlayer = self.game.getNextState(board, currentPlayer, a)
                            _, score = self.minimax(nextBoard, depth + 1, nextPlayer, alpha, beta)
                            self.memory[(s, a, currentPlayer)] = (nextBoard, nextPlayer, score)
                    else:
                        nextBoard, nextPlayer = self.game.getNextState(board, currentPlayer, a)
                        _, score = self.minimax(nextBoard, depth + 1, nextPlayer, alpha, beta)

                    if score < minEval:
                        bestMove = a
                        minEval = score
                    
                    if self.args.prune:
                        beta = np.minimum(beta, score)
                        if beta <= alpha: # Prune.
                            break
            return bestMove, minEval

