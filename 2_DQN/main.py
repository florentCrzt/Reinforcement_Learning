import numpy as np
import gym

from agent import Agent



def main():
    env =           gym.make("Breakout-v4")
    state_size =    env.observation_space.shape
    action_size =   env.action_space.n
    img_height =    110
    img_width =     84
    learning_rate = 0.001
    batch_size =    32

    breakoutAgent = Agent(action_size, img_height, img_width, learning_rate, batch_size)

    print(state_size)
    print(action_size)

    env.reset()
    observation = env.step(0)
    print(observation[0].shape)

    #Grayscale the observation
    observation_grayscale = np.dot(observation[0][...,:3],[0.3,0.6,0.1])
    print(observation_grayscale.shape)
    observation_grayscale_resize = np.resize(observation_grayscale, (110,84))
    print(observation_grayscale_resize.shape)

    action = breakoutAgent.act(observation_grayscale_resize)
    print(action)
    state, reward, done, info = env.step(action)
    print(state.shape)

if __name__ == "__main__":
    main()
