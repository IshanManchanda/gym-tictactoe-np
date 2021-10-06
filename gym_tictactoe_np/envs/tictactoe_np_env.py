import gym
import numpy as np
from gym import spaces


class TicTacToeEnv(gym.Env):
    """
    3D TicTacToe environment without safety checks

    The board is stored as a 3x3x3 numpy int_ array with player tokens.
    A value of 0 denotes an empty cell,
    1 denotes player 1 ('x'), and -1 denotes player 2 ('o').

    Actions are given by a 3-element numpy int_ array with values in {0, 1, 2}.
    The first number represents the block to move in,
    second represents the row, and third the column.
    """
    metadata = {'render.modes': ['human']}

    # Define action space, observation space, and reward range
    action_space = spaces.MultiDiscrete([3, 3, 3])
    observation_space = spaces.Box(-1, 1, shape=(3, 3, 3), dtype=np.int_)
    reward_range = (0, 1)

    # Player symbols for rendering board. 0 -> '-', 1 -> 'x', -1 -> 'o'
    symbols = '-xo'

    def __init__(self):
        super(TicTacToeEnv, self).__init__()

        # Initial state
        self.board = TicTacToeEnv.get_empty_board()
        self.current = 0
        self.round = 0
        self.done = False

    def reset(self):
        """
        Reset environment to initial state and return initial observation

        Returns
        -------
        observation : numpy.ndarray
            3x3x3 numpy int array representing the new board state
        """
        # Reset to the initial state and return board as observation
        self.board = TicTacToeEnv.get_empty_board()
        self.current = 0
        self.round = 0
        self.done = False
        return self.board

    def step(self, action):
        """
        Execute one time step within the environment

        Parameters
        ----------
        action : numpy.ndarray
            3-element numpy int array which represents the cell to play on.
            First element specifies block, second row, and third column.

        Returns
        -------
        observation : numpy.ndarray
            3x3x3 numpy int array representing the new board state
        reward : int
            Reward obtained after current move. 1 if game won else 0
        done : bool
            True if the game is over else False
        info : dict
            Additional information for debugging
        """
        self.board[action] = self.current
        self.round += 1

        # Check if game won on this move
        reward = self.check_win(action)
        # Check if game ended in a draw on this move
        self.done = reward or self.round == 27

        # Next player's turn. 0 -> 2 and 2 -> 0
        self.current = 2 - self.current

        info = {}
        # Add 1 to board to conform to gym's observation space
        return self.board + 1, reward, self.done, info

    def render(self, mode='human', close=False):
        """
        Render the environment to the screen
        """
        # Render the environment to the screen
        out = []
        for r in range(3):
            for b in range(3):
                for c in range(3):
                    out.append(TicTacToeEnv.symbols[self.board[b, r, c]])
                    out.append(' ')
                out.append('   ')
            out.append('\n')
        print(''.join(out))

    def check_win(self, action):
        """
        Checks if current action wins the game

        Parameters
        ----------
        action : numpy.ndarray
            3-element numpy int array which represents the cell to play on.
            First element specifies block, second row, and third column.

        Returns
        -------
        done : bool
            True if the current action wins the game else False
        """
        # Target sum is 3 times the current player's numerical token
        # Sum can only be equal if all 3 board values are the same as the token
        target_sum = 3 * self.current
        b, r, c = action

        # Check if win along column
        if np.sum(self.board[b, r, :]) == target_sum:
            return True
        # Check if win along row
        if np.sum(self.board[b, :, c]) == target_sum:
            return True
        # Check if win along board axis (vertical)
        if np.sum(self.board[:, r, c]) == target_sum:
            return True

        # Main Diagonal within the board
        if r == c:
            if np.sum(self.board[b, (0, 1, 2), (0, 1, 2)]) == target_sum:
                return True
        # Anti Diagonal within the board
        if r + c == 2:
            if np.sum(self.board[b, (0, 1, 2), (2, 1, 0)]) == target_sum:
                return True

        # Main Diagonal on the side (col fixed)
        if b == r:
            if np.sum(self.board[(0, 1, 2), (0, 1, 2), c]) == target_sum:
                return True
        # Anti Diagonal on the side
        if b + r == 2:
            if np.sum(self.board[(0, 1, 2), (2, 1, 0), c]) == target_sum:
                return True

        # Main Diagonal on the face (row fixed)
        if c == b:
            if np.sum(self.board[(0, 1, 2), r, (0, 1, 2)]) == target_sum:
                return True
        # Anti Diagonal on the face
        if c + b == 2:
            if np.sum(self.board[(0, 1, 2), r, (2, 1, 0)]) == target_sum:
                return True

        # Check 4 body diagonals
        if np.sum(self.board[(0, 1, 2), (0, 1, 2), (0, 1, 2)]) == target_sum:
            return True
        if np.sum(self.board[(0, 1, 2), (0, 1, 2), (2, 1, 0)]) == target_sum:
            return True
        if np.sum(self.board[(0, 1, 2), (2, 1, 0), (0, 1, 2)]) == target_sum:
            return True
        if np.sum(self.board[(2, 1, 0), (0, 1, 2), (0, 1, 2)]) == target_sum:
            return True

        return False

    @staticmethod
    def get_empty_board():
        """
        Utility function that returns an empty board

        Returns
        -------
        board : numpy.ndarray
            3x3x3 numpy int array of ones
        """
        return np.ones((3, 3, 3), dtype=np.int_)

    @staticmethod
    def get_available_actions(board):
        """
        Utility function that returns currently available moves

        Parameters
        ----------
        board : numpy.ndarray
            3x3x3 numpy int array representing the board state

        Returns
        -------
        available_actions : numpy.ndarray
            Nx3 numpy array with the N currently available actions
        """
        # 1 indicates blank in the observation space
        return np.argwhere(board == 1)
