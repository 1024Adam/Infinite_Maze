from infinite_maze import controlled_run

from infinite_maze import RIGHT
from infinite_maze import DO_NOTHING

total_number_of_games = 5
games_count = 0

class Wrapper(object):
    def __init__(self):
        controlled_run(self, 0)

    def control(self, values):
        print(values)
        # This is the function that is called by the game.
        # The values dict contains important information
        # that we will need to use to train and predict

        # Do some work here
        action = int(input())
        # Finally, return the prediction
        return action

    def gameover(self, score):
        # The game has completed. Do cleanup stuff here
        global games_count
        games_count += 1

        if games_count < total_number_of_games:
            controlled_run(self, games_count)
        else:
            return

if __name__ == '__main__':
    w = Wrapper()