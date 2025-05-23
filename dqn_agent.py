import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, optimizers

class DQNAgent:
    def __init__(self, state_shape, action_size, 
                 memory_size=10000, batch_size=64, 
                 gamma=0.95, learning_rate=0.001):
        
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Models
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            layers.LSTM(64, return_sequences=True, 
                       input_shape=self.state_shape),
            layers.LSTM(64),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state[np.newaxis, ...], verbose=0)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([x[0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])
        dones = np.array([x[4] for x in minibatch])

        targets = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.amax(next_q_values[i])
        
        self.model.fit(states, targets, verbose=0)
        
    def train(self, env, episodes, update_target_every=10):
        for e in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                self.remember(state, action, reward, next_state, done)
                self.replay()
                
                state = next_state
                total_reward += reward
            
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            if e % update_target_every == 0:
                self.update_target_model()
            
            print(f"Episode: {e+1}/{episodes}, Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.2f}")