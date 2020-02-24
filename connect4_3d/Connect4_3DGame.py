# Connect4_3DGame.py
import sys
import numpy as np
import itertools

import plotly as py
import plotly.graph_objects as go

sys.path.append('..')
from Game import Game
from .Connect4_3DLogic import Board


class Connect4_3DGame(Game):
    """
    Connect4 3D Game class implementing the alpha-zero-general Game interface.
    """

    def __init__(self, args, width=4, depth=4, height=4, win_length=4, np_pieces=None):
        super(Connect4_3DGame, self).__init__(args)
        self._base_board = Board(width, depth, height, win_length, np_pieces)

    def getInitBoard(self):
        return self._base_board.np_pieces

    def getBoardSize(self):
        return (self._base_board.width, self._base_board.depth, self._base_board.height)

    def getActionSize(self):
        return self._base_board.width * self._base_board.depth

    def getNextState(self, board, player, action):
        """Returns a copy of the board with updated move, original board is unmodified."""
        b = self._base_board.with_np_pieces(np_pieces=np.copy(board))
        b.add_stone(action, player)
        return b.np_pieces, -player

    def getValidMoves(self, board, player):
        "Any zero value in top row in a valid move"
        return self._base_board.with_np_pieces(np_pieces=board).get_valid_moves()

    def getUserFriendlyMoves(self, board, player):
        '''
        Returns a list with tuples (move, user friendly representation).
        '''

        uf_moves = []
        for i, valid in enumerate(self.getValidMoves(board, player)):
            uf_moves.append((valid, self._base_board.get_action_2d(i)))
        return uf_moves

    def getGameEnded(self, board, player):
        b = self._base_board.with_np_pieces(np_pieces=board)
        winstate = b.get_win_state()
        if winstate.is_ended:
            if winstate.winner is None: # Game is over.
                # draw has very little value.
                return 1e-4
            elif winstate.winner == player:
                return +1
            elif winstate.winner == -player:
                return -1
            else:
                raise ValueError('Unexpected winstate found: ', winstate)
        else:
            # 0 used to represent unfinished game.
            return 0

    def getCanonicalForm(self, board, player):
        # Flip player from 1 to -1
        return board * player

    def getSymmetries(self, board, pi):
        """Board is 0/90/180/270 degree rotation symmetric and mirrored symmetric"""
        boards, pis = [], []
        pi_dim = np.array(pi).reshape(self._base_board.width, self._base_board.depth)
        for k in [0, 1, 2, 3]: # Rotations of 0, 90, 180, 270 degrees.
            rotation = lambda x : np.rot90(x, k=k, axes=(0, 1))
            symmetric_board = rotation(board)
            symmetric_pi = rotation(pi_dim)

            # Add rotation to symmtries.
            boards.append(symmetric_board)
            pis.append(symmetric_pi.reshape(-1))

            # Add mirrored version of symmetry to symmetries.
            for axis in [0, 1, (0, 1), (1, 0)]:
                mirroring = lambda x : np.flip(x, axis=axis)
                mirrored_board = mirroring(symmetric_board)
                mirrored_pi = mirroring(symmetric_pi)
                mirrored_pi = mirrored_pi.reshape(-1)

                # Add mirrored version to symmetries.
                boards.append(mirrored_board)
                pis.append(mirrored_pi.reshape(-1))

        symmetries = []
        for b, p in zip(boards, pis):
            # Assert original shape equals symmetry shape when width != depth.
            if b.shape == self._base_board.np_pieces.shape:
                symmetries.append((b, p))

        return symmetries

    def getCustomSymmetries(self, board, pi):
        """Board is 0/90/180/270 degree rotation symmetric and mirrored symmetric"""
        boards, pis = [], []
        pi_dim = np.array(pi).reshape(self._base_board.width, self._base_board.depth)
        for k in [0, 1, 2, 3]: # Rotations of 0, 90, 180, 270 degrees.
            rotation = lambda x : np.rot90(x, k=k, axes=(1, 2))
            symmetric_board = rotation(board)
            symmetric_pi = rotation(pi_dim)

            # Add rotation to symmtries.
            boards.append(symmetric_board)
            pis.append(symmetric_pi.reshape(-1))

            # Add mirrored version of symmetry to symmetries.
            for axis in [1, 2, (1, 2), (2, 1)]:
                mirroring = lambda x : np.flip(x, axis=axis)
                mirrored_board = mirroring(symmetric_board)
                mirrored_pi = mirroring(symmetric_pi)
                mirrored_pi = mirrored_pi.reshape(-1)

                # Add mirrored version to symmetries.
                boards.append(mirrored_board)
                pis.append(mirrored_pi.reshape(-1))

        symmetries = []
        for b, p in zip(boards, pis):
            # Assert original shape equals symmetry shape when width != depth.
            if b.shape == self._base_board.np_pieces.shape:
                symmetries.append((b, p))

        return symmetries

    def stringRepresentation(self, board):
        return str(self._base_board.with_np_pieces(np_pieces=board))
        #return board.tostring()

    @staticmethod
    def display(board):
        '''
        ...
        '''

        # List of things to plot.
        traces = []

        # Plot rods.
        for move, (xi, yi) in enumerate(itertools.product(range(board.shape[0]), range(board.shape[1]))):
            x, y, z = Connect4_3DGame.get_cylinder(xi, yi, 0, 0.05, board.shape[2] + 0.2)
            traces.append(go.Surface(x=x, y=y, z=z,
                                     colorscale=[[0, 'gray'], [1, 'gray']],
                                     text=f'action: {move} - ({xi}, {yi})'))

        # Plot spheres.
        for xi, yi, zi in itertools.product(range(board.shape[0]), range(board.shape[1]), range(board.shape[2])):
            if board[xi, yi, zi] != 0: # Don't plot when empty.
                # Create a sphere on the board.
                x, y, z = Connect4_3DGame.get_sphere(xi, yi, zi + 0.5, 0.3)
                
                # Determine color of sphere.
                if board[xi, yi, zi] == 1: # White player color.
                    color = 'red'
                elif board[xi, yi, zi] == -1: # Black player color.
                    color = 'black'
                
                # Plot the sphere.
                traces.append(go.Surface(x=x, y=y, z=z, colorscale=[[0, color], [1, color]]))

        # Create and display figure.
        fig = go.Figure(data=traces)
        fig.update_layout(showlegend=False)
        fig.show()

    @staticmethod
    def get_cylinder(center_x, center_y, center_z, radius, height, mesh=10):
        '''
        ...
        '''

        z = np.linspace(center_z, center_z + height, mesh)

        theta = np.linspace(center_z, 2 * np.pi, mesh)
        theta_grid, z = np.meshgrid(theta, z)

        x = radius * np.cos(theta_grid) + center_x
        y = radius * np.sin(theta_grid) + center_y

        return x, y, z

    @staticmethod
    def get_sphere(center_x, center_y, center_z, diameter, mesh=10):
        '''
        ...
        '''

        u = np.linspace(0, 2 * np.pi, mesh)
        v = np.linspace(0, np.pi, mesh)

        x = diameter * np.outer(np.cos(u), np.sin(v)) + center_x
        y = diameter * np.outer(np.sin(u), np.sin(v)) + center_y
        z = diameter * np.outer(np.ones(np.size(u)), np.cos(v)) + center_z

        return x, y, z