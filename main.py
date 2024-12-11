import argparse
import random
import tensorflow as tf
from agent import Agent
from Environment import *

# Set up argparse for command-line arguments
parser = argparse.ArgumentParser()

# Model
parser.add_argument('--model', type=str, default='m1', help='Type of model')
parser.add_argument('--dueling', type=bool, default=False, help='Whether to use dueling deep q-network')
parser.add_argument('--double_q', type=bool, default=False, help='Whether to use double q-learning')

# Environment
parser.add_argument('--env_name', type=str, default='Breakout-v0', help='The name of gym environment to use')
parser.add_argument('--action_repeat', type=int, default=4, help='The number of actions to be repeated')

# Etc
parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to use GPU or not')
parser.add_argument('--gpu_fraction', type=str, default='1/1', help='GPU fraction e.g. 1/3, 2/3, 3/3')
parser.add_argument('--display', type=bool, default=False, help='Whether to display the game screen or not')
parser.add_argument('--is_train', type=bool, default=True, help='Whether to do training or testing')
parser.add_argument('--random_seed', type=int, default=123, help='Value of random seed')

# Parse arguments
args = parser.parse_args()

# Set random seed
tf.random.set_seed(args.random_seed)
random.seed(args.random_seed)

if args.gpu_fraction == '':
    raise ValueError("--gpu_fraction should be defined")

def calc_gpu_fraction(fraction_string):
    idx, num = fraction_string.split('/')
    idx, num = float(idx), float(num)

    fraction = 1 / (num - idx + 1)
    print(f" [*] GPU: {fraction:.4f}")
    return fraction

def main():
    up_lanes = [3.5 / 2, 3.5 / 2 + 3.5, 250 + 3.5 / 2, 250 + 3.5 + 3.5 / 2, 500 + 3.5 / 2, 500 + 3.5 + 3.5 / 2]
    down_lanes = [250 - 3.5 - 3.5 / 2, 250 - 3.5 / 2, 500 - 3.5 - 3.5 / 2, 500 - 3.5 / 2, 750 - 3.5 - 3.5 / 2, 750 - 3.5 / 2]
    left_lanes = [3.5 / 2, 3.5 / 2 + 3.5, 433 + 3.5 / 2, 433 + 3.5 + 3.5 / 2, 866 + 3.5 / 2, 866 + 3.5 + 3.5 / 2]
    right_lanes = [433 - 3.5 - 3.5 / 2, 433 - 3.5 / 2, 866 - 3.5 - 3.5 / 2, 866 - 3.5 / 2, 1299 - 3.5 - 3.5 / 2, 1299 - 3.5 / 2]
    width = 750
    height = 1299
    Env = Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height)
    Env.new_random_game()

    # Configure GPU options
    # This is the new TensorFlow 2.x way to configure GPU memory growth.
    gpu_fraction = calc_gpu_fraction(args.gpu_fraction)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth for each GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # Optionally, set the memory limit (if you want to limit GPU memory usage)
            tf.config.set_logical_device_configuration(
                gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=gpu_fraction)])
            print(f" [*] GPU memory limit set to {gpu_fraction:.4f}")
        except RuntimeError as e:
            print(e)

    # Start the training
    with tf.compat.v1.Session() as sess:
        agent = Agent([], Env, sess)
        if args.is_train:
            agent.train()
            agent.play()

if __name__ == '__main__':
    main()
