import tensorflow as tf
import keras
import numpy as np
import random
from collections import deque

from keras.layers import Dense, Conv2D, Flatten
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
        self.train_interval = 4
        self.update_interval = 1000
        self.total_iteration = 0
        self.game_memory = deque(maxlen=20000)
        self.model = self._build_model()
        self.target_model = self._build_model()


    def _build_model(self):
        input_layer = Input(shape=(self.img_height, self.img_width, 4))
        x = Conv2D(16, (8,8), strides=4, activation="relu")(input_layer)
        x = Conv2D(32, (4,4), strides=2, activation="relu")(x)
        x = Flatten()(x)
        x = Dense(256,activation="relu")(x)
        x = Dense(self.action_size, activation="linear")(x)
        model = Model(inputs=input_layer, outputs=x)
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        return model


    def record(self, state, action, reward, next_state, terminal):
        self.game_memory.append((state, action, reward, next_state, terminal))


    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        action = self.model.predict(state)
        return np.argmax(action[0])


    def fit(self):
        # minibatch = random.sample(self.memory, self.batch_size)
        # for state, action, reward, next_state, terminal in minibatch :
        #     target = self.model.predict(state)
        #     Q_future = self.targert_model.predict(next_state)[0]
        #     terminal_i = 1 if terminal else 0
        #     target[0][action] = reward + (1-terminal_i) * self.gamma * np.amax(Q_future)
        #     self.model.fit(state, target, epochs=1, verbose=0)
        # if self.exploration_rate > self.exploration_rate_min :
        #     self.exploration_rate *= self.exploration_rate_decay
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []
        y_batch = []
        minibatch = random.sample(self.game_memory, self.batch_size)
        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
            terminal_batch.append(data[4])

        predict_batch = []
        train_state_batch = []
        for i in range(1,32) :
            predict_batch.append(next_state_batch[i])
            train_state_batch.append(state_batch[i])

            if i%4 == 0 :
                predict_batch = np.transpose(predict_batch, (1,2,0))
                train_state_batch = np.transpose(train_state_batch, (1,2,0))

                terminal_batch = np.array(terminal_batch) + 0
                target_q_values_batch = self.target_model.predict(np.array(np.expand_dims(predict_batch, axis=0)))[0]
                y_batch = reward_batch + (1 - terminal_batch) * self.gamma * np.max(target_q_values_batch, axis=-1)
                self.model.train_on_batch(np.array(np.expand_dims(train_state_batch, axis=0)),[np.zeros((1,self.action_size))])
                predict_batch = []
                train_state_batch = []
            else :
                pass
        if self.exploration_rate > self.exploration_rate_min :
            self.exploration_rate *= self.exploration_rate_decay


    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())


    def save(self, name):
        self.model.save(name)
