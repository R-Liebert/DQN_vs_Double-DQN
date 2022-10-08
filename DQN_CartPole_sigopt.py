import os
from os import truncate
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf


import sigopt
from sigopt import Connection


def setup_connection(api_token):
    """
    Setup Sigopt connection.
    Set a experiment name.

    Arguments:
    api_token: Sigopt API token
    You can find your API token at https://sigopt.com/user/tokens
    max_episodes: int 
    Maximum number of episodes to run
    """
    
    conn = Connection(client_token=api_token)

    experiment = conn.experiments().create(
        name="DQN optimization (CartPole-v1)",
        type="offline",
        parameters=[
            dict(name='hl', type='int', bounds=dict(min=1, max=3)), # hidden layers 
            dict(name='hls', type='int', bounds=dict(min=12, max=48)), # hidden layer size
            dict(name='lr', type='double', bounds=dict(min=1e-5, max=1e-1)), # learning rate
            dict(name='bs', type='int', bounds=dict(min=16, max=64)), # batch size
            dict(name='dr', type='double', bounds=dict(min=0.9, max=0.999)), # decay rate
            dict(name='g', type='double', bounds=dict(min=0.8, max=0.99)) # gamma
            ],
            metrics=[dict(name='test_reward', objective='maximize'), dict(name='final_episode', objective='minimize')],
            parallel_bandwidth=1,
            observation_budget=60,
            )
        
    print("Explore your experiment: https://app.sigopt.com/experiment/" + experiment.id + "/analysis")
    return conn, experiment

class DQNSolver:

    def __init__(self, num_states, action_space, hidden_layers, hidden_layer_size, learning_rate, batch_size, decay_rate, gamma):
        self.epsilon = 1
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.99
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.gamma = gamma
        self.action_space = action_space
        self.memory = deque(maxlen=batch_size)

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(num_states,), activation="relu"))
        for _ in range(hidden_layers):
            self.model.add(Dense(hidden_layer_size, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(learning_rate=learning_rate, decay=decay_rate))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_space)
        return np.argmax(self.model.predict(np.atleast_2d(state))[0])

    def experience_replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, state_next, terminated in batch:
            q_update = reward
            if not terminated:
                q_update = (reward + self.gamma * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.min_epsilon, self.epsilon)


def play_game(env, max_episodes=1000, hidden_layers=1, hidden_layer_size=24, learning_rate=1e-3, batch_size=32, decay_rate=1e-3, gamma=0.95):
    observations, _ = env.reset()
    num_states = len(env.observation_space.sample())
    num_actions = env.action_space.n
    dqn_solver = DQNSolver(num_states, num_actions, hidden_layers, hidden_layer_size, learning_rate, batch_size, decay_rate, gamma)

    total_rewards = []
    for episode in range(max_episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, env.observation_space.shape[0]])
        terminated = False
        truncated = False
        step = 0
        rewards = 0
        while not terminated and not truncated:
            step += 1
            #env.render()
            action = dqn_solver.act(state)
            state_next, reward, terminated, truncated, info = env.step(action)
            state_next = np.reshape(state_next, [1, env.observation_space.shape[0]])
            dqn_solver.remember(state, action, reward, state_next, terminated)
            rewards += reward
            state = state_next
            if terminated or truncated:
                print (f"terminated after: {episode} episodes, exploration: {dqn_solver.epsilon}, score: {step}" )
                if truncated:
                    step = 500

        total_rewards.append(step)
        if step >= 100 and all(total_rewards[max(0, episode - 100):(episode + 1)] >= 195):
                print(f"Environment solved in {episode} episodes, exploration: {dqn_solver.epsilon}, score: {step}")
                return np.mean(total_rewards[-100:]), episode, dqn_solver
        if not episode == 0 and episode % 10 == 0:
            dqn_solver.experience_replay()

        total_rewards.append(step)
        print (f"Episodes: {episode}, average score last hundred episodes: {np.mean(total_rewards[-100:])}" )

    print ("Done")
    return np.mean(total_rewards[-100:]), episode, dqn_solver

def test(env, model):
    rewards = 0
    steps = 0
    done = False
    truncated = False
    observation, _ = env.reset()
    while not done and not truncated:
        action = model.act(observation)
        observation, reward, done, truncated, _= env.step(action)
        steps += 1
        rewards += reward
    
    rewards = 500 if truncated==True else rewards 

    return rewards


def main():

    sigopt_token = "UDTVDVHKBTRMWMWZOFZQIJBTCEQBTWOPDZXPVIFBSNEYPDTA" # Insert your API token here.

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
    conn, experiment = setup_connection(api_token=sigopt_token)
    

    for _ in range(experiment.observation_budget):
        value_dicts = []
        env = gym.make('CartPole-v1')
        suggestion = conn.experiments(experiment.id).suggestions().create()
        assignments = suggestion.assignments
        max_steps = 500
        env._max_episode_steps = max_steps
        max_episodes = 2000

        hidden_layers = assignments['hl']
        hidden_layer_size = assignments['hls'] 
        learning_rate = assignments['lr'] #learning rate for Adam optimizer
        batch_size = assignments['bs'] 
        decay_rate = assignments['dr'] #decay rate for Adam optimizer
        gamma = assignments['g'] #discount factor

        average_score, episodes, DQNAgent = play_game(env, max_episodes, hidden_layers, hidden_layer_size, learning_rate, batch_size, decay_rate, gamma)
        test_reward = test(env, DQNAgent)

        value_dicts = [dict(name='test_reward', value=test_reward), dict(name='final_episode', value=episodes)]
        env.close()

        conn.experiments(experiment.id).observations().create(suggestion=suggestion.id,values=value_dicts)
    
        #update experiment object
        experiment = conn.experiments(experiment.id).fetch()

    assignments = conn.experiments(experiment.id).best_assignments().fetch().data[0].assignments # get best assignments

    print("BEST ASSIGNMENTS FOUND: \n", assignments)


if __name__ == "__main__":
    main() 
