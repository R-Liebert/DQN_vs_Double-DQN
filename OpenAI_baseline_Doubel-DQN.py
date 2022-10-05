import random
import numpy as np
import gym
import imageio  # write env render to mp4
import datetime
from collections import deque
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import os

'''
Original paper: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
- DQN model with Dense layers only
- Model input is changed to take current and n previous states where n = time_steps
- Multiple states are concatenated before given to the model
- Uses target model for more stable training
- More states was shown to have better performance for CartPole env

This is the Double-DQN script from OpenAI Baseline which has been modified to work with the updated CartPole-v1 environment.

'''


class DQN:
    def __init__(
            self, 
            env, 
            memory_cap=1000,
            time_steps=3,
            gamma=0.85,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            learning_rate=0.005,
            batch_size=32,
            tau=0.125,
            hidden_layers=1,
            hidden_layer_size=24,
    ):
        self.env = env
        self.memory = deque(maxlen=memory_cap)
        self.state_shape = env.observation_space.shape
        self.time_steps = time_steps
        self.stored_states = np.zeros((self.time_steps, self.state_shape[0]))
        
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # amount of randomness in e-greedy policy
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay  # exponential decay
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers # number of hidden layers in model
        self.hidden_layer_size = hidden_layer_size # number of nodes in hidden layers
        self.batch_size = batch_size
        self.tau = tau  # target model update

        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.summaries = {}

    def create_model(self):
        """
        Create model with Dense layers only.
        
        Arguments:
        ----------
        None

        Returns:
        --------
        model: keras model
        """

        model = Sequential()
        model.add(Dense(24, input_dim=self.state_shape[0]*self.time_steps, activation="relu"))
        for i in range(self.hidden_layers):
            model.add(Dense(self.hidden_layer_size, activation="relu"))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        return model
    
    def update_states(self, new_state):
        """
        Update stored states with new state. The new state is added to the end of the stored states and the first state is removed.

        Arguments:
        ----------
        new_state: np.array, mandatory
        new state to be added to stored states

        Returns:
        --------
        None
        """

        self.stored_states = np.roll(self.stored_states, -1, axis=0)
        new_state = np.asarray(new_state)
        new_state = new_state.flatten()
        if type(new_state) == tuple:
            new_state = new_state[0]
            new_state = new_state.flatten()
        elif new_state.shape == (2,):
            new_state = new_state[0]
        self.stored_states[-1] = new_state

    def act(self, test=False): 
        """
        Choose action based on epsilon-greedy policy.

        Arguments:
        ----------
        test: bool, optional
        if True, choose action with highest Q-value

        Returns:
        --------
        action: int
        action to be taken
        """

        states = self.stored_states.reshape((1, self.state_shape[0]*self.time_steps))
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        epsilon = 0.01 if test else self.epsilon  # use epsilon = 0.01 when testing
        q_values = self.model.predict(states)[0]
        self.summaries['q_val'] = max(q_values)
        if np.random.random() < epsilon:
            return self.env.action_space.sample()  # sample random action
        return np.argmax(q_values)

    def remember(self, state, action, reward, new_state, done):
        """
        Store experience in memory.

        Arguments:
        ----------
        state: np.array, mandatory
        current state
        action: int, mandatory
        action taken
        reward: float, mandatory
        reward received
        new_state: np.array, mandatory
        new state after taking action
        done: bool, mandatory
        whether the episode is done

        Returns:
        --------
        None
        """

        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        """
        Replay memory to train the target model. Sample batch_size and memory and train model on it for one epoch. 
              
        Arguments:
        ----------
        None
        
        Returns:
        --------
        None
        """
        if len(self.memory) < self.batch_size:
            return

        samples = random.sample(self.memory, self.batch_size)
        states, action, reward, new_states, done = map(np.asarray, zip(*samples))
        batch_states = np.array(states).reshape(self.batch_size, -1)
        batch_new_states = np.array(new_states).reshape(self.batch_size, -1)
        batch_target = self.target_model.predict(batch_states)
        q_future = self.target_model.predict(batch_new_states).max(axis=1)
        batch_target[range(self.batch_size), action] = reward + (1 - done) * q_future * self.gamma
        hist = self.model.fit(batch_states, batch_target, epochs=1, verbose=0)
        self.summaries['loss'] = np.mean(hist.history['loss'])

    def target_update(self):
        """
        Update target model weights with model weights.

        Arguments:
        ----------
        None

        Returns:
        --------
        None
        """

        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):  # set tau of target model to be new weights
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        """Save model to file, including weights and optimizer state. Give filename with extension .h5

        Arguments:
        ----------
        fn: str, mandatory
        filename with .h5 extension

        Returns:
        --------
        None
        """

        self.model.save(fn)

    def load_model(self, fn):
        """Load model from file, give file name with .h5 extension
        
        Arguments:
        ----------
        fn: str, mandatory
        filename with .h5 extension
        
        Returns:
        --------
        None
        """

        self.model = tf.keras.models.load_model(fn)
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

    def train(self, max_episodes=10, max_steps=500, save_freq=10):
        """Here we train the agent with the DQN algorithm. 
        We first initialize the target model with the same weights as the model.
        Then we loop over episodes and steps. In each step, we first act, then remember the experience,
        then replay the experience, and finally update the target model.

        During training, we save the model every save_freq episodes. And when done with each episode,
        we print the episode number, the total reward and log the summaries.
        
        Parameters
        ----------
        max_episodes : int, optional
        Maximum number of episodes to train for, by default 10
        max_steps : int, optional
        Maximum number of steps per episode, by default 500
        save_freq : int, optional
        Frequency of saving the model, by default 10
            
        Returns
        -------
        None

        """
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = f'logs/DQN_basic_time_step{self.time_steps}/{current_time}'
        summary_writer = tf.summary.create_file_writer(train_log_dir)

        done, episode, steps, epoch, total_reward = True, 0, 0, 0, 0
        while episode < max_episodes:
            if steps >= max_steps:
                print(f"episode {episode}, reached max steps")
                self.save_model(f"dqn_basic_maxed_episode{episode}_time_step{self.time_steps}.h5")
                break

            if done:
                with summary_writer.as_default():
                    tf.summary.scalar('Main/episode_reward', total_reward, step=episode)
                    tf.summary.scalar('Main/episode_steps', steps, step=episode)

                self.stored_states = np.zeros((self.time_steps, self.state_shape[0]))
                print(f"episode {episode}: {total_reward} reward")

                #if episode % save_freq == 0:  # save model every n episodes
                #    self.save_model(f"dqn_basic_episode{episode}_time_step{self.time_steps}.h5")

                done, cur_state, steps, total_reward = False, self.env.reset(), 0, 0
                self.update_states(cur_state)  # update stored states
                episode += 1

            action = self.act()  # model determine action, states taken from self.stored_states
            new_state, reward, done, _, _ = self.env.step(action)  # perform action on env
            # modified_reward = 1 - abs(new_state[2] / (np.pi / 2))  # modified for CartPole env, reward based on angle
            prev_stored_states = self.stored_states
            self.update_states(new_state)  # update stored states
            self.remember(prev_stored_states, action, reward, self.stored_states, done)  # add to memory
            self.replay()  # iterates default (prediction) model through memory replay
            self.target_update()  # iterates target model

            total_reward += reward
            steps += 1
            epoch += 1

            # Tensorboard update
            with summary_writer.as_default():
                if len(self.memory) > self.batch_size:
                    tf.summary.scalar('Stats/loss', self.summaries['loss'], step=epoch)
                tf.summary.scalar('Stats/q_val', self.summaries['q_val'], step=epoch)
                tf.summary.scalar('Main/step_reward', reward, step=epoch)

            summary_writer.flush()

        self.save_model(f"dqn_basic_final_episode{episode}_time_step{self.time_steps}.h5")

    def test(self, render=False, fps=30, filename='test_render.mp4'):
        """
        Test the agent in the environment. We first reset the environment, then loop over steps.
        In each step, we first act, then update the stored states.
        If render is True, we also render the environment. WHich currently don't work for unknown 
        reasons, best guess is within imageio library.

        Parameters
        ----------
        render : bool, optional
        Whether to render the environment, by default True
        fps : int, optional
        Frames per second for rendering, by default 30
        filename : str, optional
        Filename for saving the rendering, by default 'test_render.mp4'

        Returns
        -------
        rewards : float
        Total reward for the episode
        """

        done, steps, total_reward = False, 0, 0
        self.stored_states = np.zeros((self.time_steps, self.state_shape[0]))
        cur_state = self.env.reset()
        self.update_states(cur_state)
        cur_state, _, done, rewards = self.env.reset(), False, 0
        video = imageio.get_writer(filename, fps=fps)
        while not done:
            action = self.act(test=True)
            new_state, reward, done, _, _ = self.env.step(action)
            self.update_states(new_state)
            rewards += reward
            if render:
                video.append_data(self.env.render())
        video.close()
        return rewards

def main():
    """
    Here we initialize the environment, agent and train the agent.
    If you want to load a model, uncomment the load_model line.
    If you have GPU's, you're a lucky bitch, and can uncomment the GPU line
    """

    #os.environ["CUDA_VISIBLE_DEVICES"]="0" # use GPU with ID=0 (uncomment if using GPU)
    
    env = gym.make('CartPole-v1')
    env._max_episode_steps = 500
    dqn_agent = DQN(env, time_steps=10)
    dqn_agent.train(max_episodes=5)
    # dqn_agent.load_model("basic_models/time_step4/dqn_basic_episode50_time_step4.h5")
    rewards = dqn_agent.test(render=False) # For some reason render=True doesn't work
    print(f"Total rewards: {rewards}. Take a look at tensorboard for more info.")

if __name__ == "__main__":
    main()