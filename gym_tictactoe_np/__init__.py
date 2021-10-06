from gym.envs.registration import register

register(
    id='tictactoe-np-v0',
    entry_point='gym_tictactoe_np.envs:TicTacToeEnv',
)
