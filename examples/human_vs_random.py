from random import shuffle

from agents.agent_human import HumanAgent
from agents.agent_random import RandomAgent
from gym_tictactoe_np.envs.tictactoe_np_env import TicTacToeEnv


def main():
    # Driver code to run human-vs-RandomAgent 3D TicTacToe
    # Create environment
    env = TicTacToeEnv()

    # Assign player 1 and 2 randomly to human and agent
    players = [HumanAgent(), RandomAgent()]
    shuffle(players)
    if isinstance(players[0], HumanAgent):
        print("Human is player 1 (x), machine is player 2 (o)")
    else:
        print("Machine is player 1 (x), human is player 2 (o)")

    # Counter for moves
    moves, reward, done = 0, 0, False
    state = env.reset()

    print("==== Starting Human vs Machine Game ====\n")
    while not done:
        print(f"Move {moves + 1}")

        # Get the player to move
        player = players[moves % 2]

        # Get chosen action
        action = player.act(state)

        # Check if human wants to quit
        if action is None:
            print("==== Exiting ====")
            break

        # Perform the move and render the board
        state, reward, done, info = env.step(action)
        env.render()

        moves += 1

    # Game over, check if ended in draw
    if reward == 0:
        print("==== Finished: Game ended in draw ====")
    # Game was won by last player, print
    else:
        winner = 1 + (moves + 1) % 2
        print(f"==== Finished: Game won by player {winner}! ====")


if __name__ == '__main__':
    main()
