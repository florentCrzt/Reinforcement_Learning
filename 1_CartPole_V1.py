import gym
import tensorflow as tf
import numpy as np
import os, sys, glob
import random
import keras
from collections import deque

from keras.layers import Dense
from keras.engine.input_layer import Input
from keras.models import Model


class CartPoleAgent:
    def __init__(self,state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.exploration_rate = 1.0
        self.exploration_rate_decay = 0.99
        self.exploration_rate_min = 0.01
        self.batch_size = 64
        self.memory = deque(maxlen=20000)
        self.gamma = 0.95
        self.model = self._build_model()
        self.target_model = self._build_model()

    def _build_model(self):
        input_layer = Input(shape=(self.state_size,))
        x = Dense(64, activation="relu")(input_layer)
        x = Dense(64, activation="relu")(x)
        x = Dense(self.action_size, activation="linear")(x)
        model = Model(inputs=input_layer, outputs=x)
        model.compile(loss="mean_squared_error", optimizer="Adam")
        return model

    def act(self, state):
        if np.random.rand() <= self.exploration_rate :
            return random.randrange(self.action_size)
        action = self.model.predict(state)
        return np.argmax(action[0])

    def fit(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch :
            target = self.model.predict(state)
            if done :
                target[0][action] = reward
            else :
                Q_future = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(Q_future)
            self.model.fit(state,target, epochs=1,verbose=0)
        if self.exploration_rate > self.exploration_rate_min :
            self.exploration_rate *= self.exploration_rate_decay

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def record(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def save(self):
        self.model.save("CartPole.h5")

def training():
    env = gym.make("CartPole-v0")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    cart_pole = CartPoleAgent(state_size, action_size)

    EPISODES = 1000
    GOAL_STEP = 250
    BATCH_SIZE = 64

    for ep in range(EPISODES):
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for i in range(GOAL_STEP):
            env.render()
            action = cart_pole.act(state)
            next_state, reward, done, info = env.step(action)
            reward += 1
            if done :
                reward += 1
                cart_pole.update_target_model()
                break

            next_state = np.reshape(next_state, [1, state_size])
            cart_pole.record(state, action, reward, next_state, done)

            if len(cart_pole.memory) >= BATCH_SIZE :
                cart_pole.fit()

            # if abs(next_state[0][0]) >2.4 :
            #     break
            # if abs(next_state[0][2]) >0.209 :
            #     break

            state = next_state
            score += reward
        print("episode : {}/{}, score: {}, exploration rate : {:.2}".format(ep, EPISODES, score, cart_pole.exploration_rate))

if __name__ == "__main__":
    training()
