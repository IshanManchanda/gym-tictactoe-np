from gym.envs.registration import register

from gym_tictactoe_np.envs import TicTacToeEnv

register(
    id='tictactoe-np-v0',
    entry_point='gym_tictactoe_np.envs:TicTacToeEnv',
)
