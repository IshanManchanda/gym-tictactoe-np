import numpy as np

from gym_tictactoe_np import TicTacToeEnv


class HumanAgent:
    # HumanAgent class
    def act(self, board):
        available_actions = TicTacToeEnv.get_available_actions(board)
        action = None

        # Loop until valid input
        while True:
            # Get user input and check for quit signal
            inp = input('Enter position [000 - 222], q to quit: ')
            if inp.lower() == 'q':
                return None

            # Continue iterating if input invalid, otherwise break and return
            if len(inp) != 3:
                print(f"Invalid input: '{inp}'")
                continue

            # Try converting input string into action
            try:
                action = np.array([x for x in inp], dtype=np.int_)
            except ValueError:
                print(f"Invalid input: '{inp}'")
                continue

            # Check if action present in available actions
            try:
                if not np.any(np.all(action == available_actions, axis=1)):
                    raise ValueError
            except (ValueError, np.AxisError):
                print(f"Illegal position: '{inp}'")
                continue

            break

        return action
