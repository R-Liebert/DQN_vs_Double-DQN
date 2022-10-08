import random
import datetime
import numpy as np
from gym import wrappers

from collections import deque
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import gym
from gym import wrappers

import os


'''
Original paper: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
- DQN model with Dense layers only
- Model input is changed to take current and n previous states where n = time_steps
- Multiple states are concatenated before given to the model
- Uses target model for more stable training
- More states was shown to have better performance for CartPole env

This is the Double-DQN script from OpenAI Baseline which has been modified to work with the updated CartPole-v0 environment.
And edited to be a vanilla DQN. Sigopt is used to tune the hyperparameters.

'''


class MyModel(tf.keras.Model):
    def __init__(self, num_states, hidden_units, hidden_layers, num_actions, lr):
        super(MyModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))
        self.hidden_layers = []
        for _ in range(hidden_layers):
            self.hidden_layers.append(tf.keras.layers.Dense(
                hidden_units, activation='relu', kernel_initializer='RandomNormal'))
        self.output_layer = tf.keras.layers.Dense(
            num_actions, activation='linear', kernel_initializer='RandomNormal')
        
    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output


class DQN:
    def __init__(self, num_states, num_actions, hidden_units, hidden_layers, gamma, max_experiences, min_experiences, batch_size, lr, max_steps, decay_rate):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(learning_rate=lr, decay=decay_rate)
        self.gamma = gamma
        self.model = MyModel(num_states, hidden_units, hidden_layers, num_actions, lr)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.max_steps = max_steps

    def predict(self, inputs):
        return self.model(np.atleast_2d(inputs.astype('float32')))


    def get_action(self, states, epsilon):
        valid_actions = [a for a in range(self.num_actions)] 
        if np.random.random() < epsilon:
            return np.random.choice(valid_actions)
        else:
            best_action = np.argmax(self.predict(np.atleast_2d(states))[0])
            if best_action in valid_actions:
                return best_action
            else:
                return np.random.choice(valid_actions)

    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)


def play_game(env, TrainNet, epsilon, copy_step):
    rewards = 0
    iter = 0
    done = False
    truncated = False
    observations, _ = env.reset()
    while not done and not truncated:
        action = TrainNet.get_action(observations, epsilon)
        prev_observations = observations
        observations, reward, done, truncated, _ = env.step(action)
        rewards += reward
        if done or truncated:
            env.reset()
            if truncated:
                rewards = -500

        exp = {'s': prev_observations, 'a': action, 'r': reward, 's2': observations, 'done': done}
        TrainNet.add_experience(exp)


    return rewards

def test(env, TrainNet):
    rewards = 0
    steps = 0
    done = False
    truncated = False
    observations, _ = env.reset()
    while not done and not truncated:
        action = TrainNet.get_action(observations, 0)
        observations, reward, done, truncated, _= env.step(action)
        steps += 1
        rewards += reward
    
    rewards = -500 if truncated==True else rewards 

    return rewards


def main():
    """
    Here we initialize the environment, agent and train the agent.
    If you want to load a model, uncomment the load_model line.
    If you have GPU's, you're a lucky bitch, and can uncomment the GPU line
    
    """

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    os.environ["CUDA_VISIBLE_DEVICES"]="0"  # use GPU with ID=0 (uncomment if GPU is available)



    env = gym.make('MountainCar-v0')


    max_steps = 500 # Environment max step
    env._max_episode_steps = max_steps

    gamma = 0.95 
    copy_step = 25
    num_states = 2 # hardcoded until solved
    num_actions = 3 # hardcoded until solved
    hidden_units = 2
    hidden_layers = 24
    max_experiences = 10000
    min_experiences = 100
    batch_size = 32
    lr = 1e-3
    decay_rate = 0.95
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/dqn/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)
    TrainNet = DQN(num_states, num_actions, hidden_units, hidden_layers, gamma, max_experiences, min_experiences, batch_size, lr, max_steps, decay_rate)
    max_episodes = 1000
    total_rewards = np.empty(max_episodes)
    epsilon = 1
    decay = 0.99
    min_epsilon = 0.1
    for episode in range(max_episodes):
        epsilon = max(min_epsilon, epsilon * decay)
        total_reward = play_game(env, TrainNet, epsilon, copy_step)
        total_rewards[episode]  = total_reward
        avg_rewards = np.mean(total_rewards[max(0, episode - 100):(episode + 1)])
        with summary_writer.as_default():
            tf.summary.scalar('episode reward', total_reward, step=episode)
            tf.summary.scalar('running avg reward(100)', avg_rewards, step=episode)
        if episode % 100 == 0 and episode != 0:
            print(f"episode: {episode}, episode reward: {total_reward}, eps: {epsilon}, avg reward (last 100): {avg_rewards}")
        # Check if last 100 episodes have total_reward >= 195 to approve training
        if episode >= 100 and all(total_rewards[max(0, episode - 100):(episode + 1)] > -100):
            final_episode = episode
            print(f"You solved it in {final_episode} episodes!")
            break
        
    # if final_episode doesn't exist set too max episodes.
    final_episode = final_episode if 'final_episode' in locals() else max_episodes

    test_reward = test(env, TrainNet)
    
    print(f"Test reward: {test_reward}, avgerage reward last 100 episodes: {avg_rewards}, num episodes: {final_episode}")
    env.close()



if __name__ == "__main__":
    main() 
   
