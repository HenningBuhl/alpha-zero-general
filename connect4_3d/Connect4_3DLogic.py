from collections import namedtuple
import numpy as np

WinState = namedtuple('WinState', 'is_ended winner')


class Board():
    """
    Connect4 3D Board.
    """

    def __init__(self, width=4, depth=4, height=4, win_length=4, np_pieces=None):
        "Set up initial board configuration."
        self.width = width
        self.depth = depth
        self.height = height
        self.win_length = win_length
        self.find = np.ones(shape=(self.win_length))

        if np_pieces is None:
            self.np_pieces = np.zeros([self.width, self.depth, self.height])
        else:
            self.np_pieces = np_pieces
            assert self.np_pieces.shape == (self.width, self.depth, self.height)

    def add_stone(self, action, player):
        "Create copy of board containing new stone."
        x, y = self.get_action_2d(action)
        available_idx, = np.where(self.np_pieces[x, y, :] == 0)
        if len(available_idx) == 0:
            raise ValueError("Can't play move %s on board %s" % (action, self))

        #self.np_pieces[available_idx[-1]][column] = player # Original Code.
        self.np_pieces[x, y, available_idx[0]] = player

    def get_valid_moves(self):
        "Any zero value in top plane is a valid move"
        return self.np_pieces[:, :, -1].reshape(-1) == 0

    def get_win_state(self):        
        # Check if a player has won the game.
        for player in [1, -1]:
            player_pieces = self.np_pieces == player
            if self.game_won(player_pieces, self.find):
                return WinState(True, player)

        # Check if game is drawn
        if not self.get_valid_moves().any():
            return WinState(True, None)
        
        # Game is not ended yet.
        return WinState(False, None)

    def with_np_pieces(self, np_pieces):
        """Create copy of board with specified pieces."""
        if np_pieces is None:
            np_pieces = self.np_pieces
        return Board(self.width, self.depth, self.height, self.win_length, np_pieces)

    def get_action_1d(self, action_2d):
        y, z = action_2d
        action_1d = y * self.width + z
        #print(f'{action_2d} converted to {action_1d}')
        return action_1d
    
    def get_action_2d(self, action_1d):
        y = action_1d // self.width
        z = action_1d % self.width
        action_2d = (y, z)
        #print(f'{action_1d} converted to {action_2d}')
        return action_2d

    def get_sub_boards(self, board):
        '''
        Returns all 2D sub boards.

        Args:
            board: A 3D numpy array.
        
        Returns:
            All 2D sub boards.
        '''

        sub_boards = []
        for x in range(board.shape[0]):
            sub_boards.append(board[x, :, :])

        for y in range(board.shape[1]):
            sub_boards.append(board[:, y, :])

        for z in range(board.shape[2]):
            sub_boards.append(board[:, :, z])

        for b in [board]:#, board[::-1,::,::], board[::,::-1,::], board[::,::,::-1]]:
            for a1, a2 in [(0,1), (1,2), (2,0)]:
                for o in range(-b.shape[a1]+1, b.shape[a2]):
                    diag = np.diagonal(b, offset=o, axis1=a1, axis2=a2)
                    sub_boards.append(diag)
        return sub_boards

    def get_lines(self, board):
        '''
        Returns all 1D sub boards.
        
        Args:
            board: A 2D numpy array.
        
        Returns:
            All 1D sub boards.
        '''
        
        lines = []
        for x in range(board.shape[0]): # Horizontal.
            lines.append(board[x, :])

        for y in range(board.shape[1]): # Vertical.
            lines.append(board[:, y])

        for b in [board[::, ::], board[::-1, ::]]: # Diagonal.
            diag = [b.diagonal(i) for i in range(-b.shape[0] + 1, b.shape[1])]
            lines.extend(diag)
        return lines

    def get_sub_lines(self, line, find):
        '''
        Returns all sub lines.

        Args:
            line: A 1D numpy array.
            find: The sequence to find.
        
        Returns:
            All sub lines.
        '''

        n = len(find)
        sub_lines = np.fromfunction(lambda i, j: line[i + j],
                                    (len(line) - n + 1, n), dtype=np.int)    
        return sub_lines

    def get_valid_lines(self, lines, find):
        '''
        Returns lines which can contain the sequence.

        Args:
            line: A 1D numpy array.
            find: The sequence to find.
        
        Returns:
            Valid lines.
        '''

        lines = [line for line in lines if len(line) >= len(find)]
        return lines

    def game_won(self, board, find):
        '''
        Determines whether a sequence is contained in the board.

        Args:
            board: Any board (1D, 2D, 3D).
            find: The sequence to find.
        
        Returns:
            Whether the sequence was found.
        '''

        #board = np.squeeze(board)
        rank = len(board.shape)
        if rank == 1:
            return self.game_won_1d(board, find)
        elif rank == 2:
            return self.game_won_2d(board, find)
        elif rank == 3:
            return self.game_won_3d(board, find)
        else:
            raise ValueError('Board rank must be 1, 2 or 3.')

    def game_won_1d(self, board, find):
        if len(find) > len(board): return False
        for sub_line in self.get_sub_lines(board, find):
            if np.array_equal(find, sub_line): # Look for normal sequence.
                return True
            #if np.array_equal(find[::-1], sub_line): # Look for reversed sequence.
            #    return True
        return False

    def game_won_2d(self, board, find):
        for line in self.get_lines(board):
            if self.game_won_1d(line, find):
                return True
        return False

    def game_won_3d(self, board, find):
        for sub_board in self.get_sub_boards(board):
            if self.game_won_2d(sub_board, find):
                return True
        return False

    def __str__(self):
        return str(self.np_pieces)