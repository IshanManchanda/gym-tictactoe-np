import gym

from gym_tictactoe_np.agents import RandomAgent


def main():
    # Driver code to run human-vs-RandomAgent 3D TicTacToe
    # Create environment
    env = gym.make('tictactoe-np-v0')

    # Assign player 1 and 2 randomly to human and agent
    players = [RandomAgent(), RandomAgent()]

    # Counter for moves
    moves, reward, done = 0, 0, False
    state = env.reset()

    print("==== Starting Machine vs Machine Game ====\n")
    while not done:
        print(f"Move {moves + 1}")
        # Get the player to move
        player = players[moves % 2]

        # Get chosen action
        action = player.act(state)

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
