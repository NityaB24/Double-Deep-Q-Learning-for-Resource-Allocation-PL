from __future__ import division, print_function
import numpy as np 
from Environment import *
import matplotlib.pyplot as plt

# This py file uses the random algorithm.

def main():
    up_lanes = [3.5/2, 3.5/2 + 3.5, 250 + 3.5/2, 250 + 3.5 + 3.5/2, 500 + 3.5/2, 500 + 3.5 + 3.5/2]
    down_lanes = [250 - 3.5 - 3.5/2, 250 - 3.5/2, 500 - 3.5 - 3.5/2, 500 - 3.5/2, 750 - 3.5 - 3.5/2, 750 - 3.5/2]
    left_lanes = [3.5/2, 3.5/2 + 3.5, 433 + 3.5/2, 433 + 3.5 + 3.5/2, 866 + 3.5/2, 866 + 3.5 + 3.5/2]
    right_lanes = [433 - 3.5 - 3.5/2, 433 - 3.5/2, 866 - 3.5 - 3.5/2, 866 - 3.5/2, 1299 - 3.5 - 3.5/2, 1299 - 3.5/2]
    width = 750
    height = 1299

    # Initialize a list to hold sum V2I rate data for each number of vehicles
    sum_v2i_rates = []
    
    n_values = [20, 40, 60, 80, 100]  # Different numbers of vehicles
    number_of_game = 50
    n_step = 100

    # Create the environment
    Env = Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height)

    # Run simulations for each number of vehicles
    for n in n_values:
        V2I_Rate_List = np.zeros([number_of_game, n_step])
        print(f"Running simulation for {n} vehicles")

        for game_idx in range(number_of_game):
            Env.new_random_game(n)
            for i in range(n_step):
                actions = np.random.randint(0, 20, [n, 3])
                power_selection = np.zeros(actions.shape, dtype='int')
                actions = np.concatenate((actions[..., np.newaxis], power_selection[..., np.newaxis]), axis=2)
                reward, _ = Env.act(actions)
                V2I_Rate_List[game_idx, i] = np.sum(reward)

        # Calculate the mean sum V2I rate over all games and steps
        sum_v2i_rate = np.mean(V2I_Rate_List)
        sum_v2i_rates.append(sum_v2i_rate)

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.plot(n_values, sum_v2i_rates, marker='o', linestyle='-', color='b')
    plt.title('Number of Vehicles vs Sum Rate of V2I Links')
    plt.xlabel('Number of Vehicles')
    plt.ylabel('Sum Rate of V2I Links (Mb/s)')
    plt.grid(True)
    plt.xlim(10, 110)  # Set the x-axis limits to encompass the range of n_values
    plt.show()

if __name__ == "__main__":
    main()
