import gym
import tensorflow as tf
import keras
import numpy as np
import random

from collections import deque

from keras.layers import Dense
from keras.engine.input_layer import Input
from keras.models import Model


# https://github.com/openai/gym/wiki/MountainCar-v0



def random_agent() :
    for episode in range(5):
        print("##############################EPISODE ",episode,"############################")
        env.reset()
        for t in range(200):
            env.render()
            action = env.action_space.sample() # return a random value in the space action value
            observation, reward, done, info = env.step(action) # do the action
            print("action : ", action, "| obs : ", observation, " | reward : ", reward, " | info :", info)
            if done :
                break




class CarAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.exploration_rate = 1.0
        self.exploration_rate_decay = 0.95
        self.exploration_rate_min = 0.01
        self.batch_size = 64
        self.memory = deque(maxlen=20000)
        self.gamma = 0.95
        self.model = self._build_model()
        self.target_model = self._build_model()


    def _build_model(self):
        input_layer = Input(shape=(2,))
        x = Dense(64, activation="relu")(input_layer)
        x = Dense(64, activation="relu")(x)
        x = Dense(self.action_size, activation="linear")(x)
        model = Model(inputs=input_layer, outputs=x)
        model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(lr=0.001))
        model.summary()
        return model


    def act(self, state):
        if np.random.rand() <= self.exploration_rate :
            return random.randrange(self.action_size)
        action = self.model.predict(state)
        return np.argmax(action[0])


    def record(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch :
            target = self.model.predict(state)
            if done :
                target[0][action] = reward
            else :
                Q_future = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(Q_future)
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.exploration_rate > self.exploration_rate_min :
            self.exploration_rate *= self.exploration_rate_decay

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def save(self, name):
        self.model.save(name)



def main():
    env = gym.make('MountainCar-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print("State : ", state_size)
    print("Action : ", action_size)
    agent = CarAgent(state_size, action_size)


    episodes = 60

    goal_step = 200
    score_requirement = -198

    done = False
    batch_size = 64

    
    for ep in range(episodes):
        scores = 0
        state = env.reset() # set the initial state
        state = np.reshape(state, [1, state_size])

        for time in range(200):

            env.render()
            action = agent.act(state) # determine the action
            next_state, reward, done, info = env.step(action) # execute it

            #Reward the action
            if next_state[1] > state[0][1] and next_state[1]>0 and state[0][1] > 0 :
                reward += 15
            elif next_state[1] < state[0][1] and next_state[1]<0 and state[0][1] < 0 :
                reward += 15

            if done :
                reward += 10000
                agent.update_target_model()
                break
            else :
                reward -= 10

            next_state = np.reshape(next_state, [1, state_size])
            #Save the information for training the model
            agent.record(state, action, reward, next_state, done)
            #Set the state for the next step
            state = next_state
            scores += reward

            if len(agent.memory) > batch_size :
                agent.replay()

            print("episode : {}/{}, score: {}, exploration rate : {:.2}".format(ep, episodes, scores, agent.exploration_rate))

    agent.save("model1.h5")

if __name__ == "__main__":
    main()
