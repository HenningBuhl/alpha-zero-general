import numpy as np
from collections import deque

class Game():
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.

    See othello/OthelloGame.py for an example implementation.
    """
    def __init__(self, args):
        self.args = args

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        pass

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        pass

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        pass

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        pass

    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        pass

    def getUserFriendlyMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player
        
        Returns:
            a list with tuples (move, user friendly representation).
        """

        uf_moves = []
        for i, valid in enumerate(self.getValidMoves(board, player)):
            uf_moves.append((valid, (int(i/self.n), int(i%self.n))))
        return uf_moves
    
    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               
        """
        pass

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        pass

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        pass
    
    def getCustomSymmetries(self, customInput, pi):
        pass

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        pass

    def getCustomInputShape(self):
        _, dummy_input = self.getCustomInput(np.zeros(self.getBoardSize()), 1)
        return dummy_input.shape

    def getCustomInput(self, cannonicalBoard, curPlayer, boardHistory=None, prevCannonicalView=None):
        '''
        Standard implementation of custom input construction with:
            - numBoardHistory
            - splitPlayerPieces
            - reverseBoardOrder
            - rotatePlayerBoard
            - usePlayerChannel
        Functional due to thread and process safety.

        Args:
            cannonicalBoard: The current cannonical board.
            curPlayer: The current player.
            boardHistory: The previous states of the board from the absolute view point (default None).
            prevCannonicalView: The previous cannonical view.

        Returns:
            A tuple containing (boardHistory, cannonicalView)
        '''
        # Convert cannonicalBoard into absolute board view.
        board = cannonicalBoard * curPlayer

        # First move of the game.
        if boardHistory is None or len(boardHistory) == 0:
            numPieces = 2 if self.args.splitPlayerPieces else 1
            numChannels = (self.args.numBoardHistory + 1) * numPieces + self.args.usePlayerChannel # numBoardHistory=0 means only 2 channes + an additional channel for curPlayerChannel.
            custom_input_shape = (numChannels, *self.getBoardSize()) # Custom input shape.
            boardHistory = deque([np.zeros(self.getBoardSize()) for _ in range(self.args.numBoardHistory+1)], maxlen=self.args.numBoardHistory+1)
        else:
            custom_input_shape = prevCannonicalView.shape

        # Add new board to boardHistory deque.
        newBoardHistory = boardHistory.copy()
        newBoardHistory.appendleft(board)

        # Create cannonicalView.
        cannonicalView = []

        # Create cannonical history.
        cannonicalHistory = [b * (1 if self.args.usePlayerChannel else curPlayer) for b in newBoardHistory]

        # Add board channels to cannonicalView.
        for b in cannonicalHistory:
            if self.args.splitPlayerPieces:
                players = [1, -1] if self.args.usePlayerChannel else [curPlayer, -curPlayer]
                for piece in players:
                    pieceChannel = np.array(b == piece) # Convert bool to float? Print dtype of b...
                    cannonicalView.append(pieceChannel)
            else:
                cannonicalView.append(self.getCanonicalForm(b, curPlayer))

        # Rotate board 180° for player -1.
        if self.args.rotatePlayerBoard and curPlayer == -1:
            cannonicalView = [np.rot90(b, k=2) for b in cannonicalView]
        
        # Reverse board order.
        if self.args.reverseBoardOrder:
            cannonicalView = cannonicalView[::-1]

        # Add playerChannel.
        if self.args.usePlayerChannel:
            curPlayerChannel = np.full(self.getBoardSize(), int(curPlayer > 0))
            cannonicalView.append(curPlayerChannel)
        
        # Construct cannonicalView.
        cannonicalView = np.stack(cannonicalView, axis=0)

        assert cannonicalView.shape == custom_input_shape
        return newBoardHistory, cannonicalView

