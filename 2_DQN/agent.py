import tensorflow as tf
import keras
import numpy as np
import random
from collections import deque

from keras.layers import Dense, Conv2D
from keras.engine.input_layer import Input
from keras.models import Model
from keras.optimizers import Adam

class Agent:
    def __init__(self, action_size, height, width, batch_size):
        self.action_size = action_size
        self.height = height
        self.width = width
        self.exploration_rate = 1.0
        self.exploration_rate_decay = 0.95
        self.exploration_rate_min = 0.001
        self.gamma = 0.95
        self.batch_size = batch_size
        self.memory = deque(maxlen=20000)
        self.model = self._build_model()
        self.target_model = self._build_model()


    def _build_model(self):
        input_layer = Input(shape=(self.height, self.width, 4))
        x = Conv2D(16, (8,8), stride=4, activation="relu")(input_layer)
        x = Cond2D(32, (4,4), stride=2, activation="relu")(x)
        x = Dense(256,activation="relu")(x)
        x = Dense(self.action_size, activation="linear")(x)
        model = Model(inputs=input_layer, outputs=x)
        model.compile(loss="mean_squared_error", optimizer=Adam())
        return model


    def record(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))


    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        action = self.model.predict(state)
        return np.argmax(action[0])


    def fit(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state in minibatch :
            target = self.model.predict(state)



    def save(self, name):
