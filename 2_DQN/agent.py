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
    def __init__(self, action_size, height, width, lr, batch_size):
        self.action_size = action_size
        self.img_height = height
        self.img_width = width
        self.exploration_rate = 1.0
        self.exploration_rate_decay = 0.95
        self.exploration_rate_min = 0.001
        self.learning_rate = lr
        self.gamma = 0.95
        self.batch_size = batch_size
        self.memory = deque(maxlen=20000)
        self.model = self._build_model()
        self.target_model = self._build_model()


    def _build_model(self):
        input_layer = Input(shape=(self.img_height, self.img_width, 4))
        x = Conv2D(16, (8,8), strides=4, activation="relu")(input_layer)
        x = Conv2D(32, (4,4), strides=2, activation="relu")(x)
        x = Dense(256,activation="relu")(x)
        x = Dense(self.action_size, activation="linear")(x)
        model = Model(inputs=input_layer, outputs=x)
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        return model


    def record(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state, terminal))


    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        action = self.model.predict(state)
        return np.argmax(action[0])


    def fit(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, terminal in minibatch :
            target = self.model.predict(state)
            Q_future = self.targert_model.predict(next_state)[0]
            terminal_i = 1 if terminal else 0
            target[0][action] = reward + (1-terminal_i) * self.gamma * np.amax(Q_future)
            self.model.fit(state, target, epochs=1, verbose=0)
            if self.exploration_rate > self.exploration_rate_min :
                self.exploration_rate *= self.exploration_rate_decay


    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())


    def save(self, name):
        self.model.save(name)
