import gym
import tensorflow as tf
import keras
import numpy as np
import random

from collections import deque

from keras.layers import Dense
from keras.engine.input_layer import Input
from keras.models import Model, load_model

env = gym.make('MountainCar-v0')
agent = load_model("model1.h5")
state_size = env.observation_space.shape[0]
steps= []

for i in range(100):
    step = 0
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(200):
        env.render()
        action = agent.predict(state)
        next_state, reward, done, info = env.step(np.argmax(action[0]))
        state = np.reshape(next_state, [1,state_size])
        step += 1
        if done :
            break
    print(step)
    steps.append(step)
   
print("AVG STEP : ", np.sum(steps)/len(steps))