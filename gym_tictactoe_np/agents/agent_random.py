import numpy as np

from gym_tictactoe_np import TicTacToeEnv


class RandomAgent:
    # RandomAgent class
    def act(self, board):
        # Sample one action randomly from all available actions
        available_actions = TicTacToeEnv.get_available_actions(board)
        idx = np.random.choice(len(available_actions))
        return available_actions[idx]
