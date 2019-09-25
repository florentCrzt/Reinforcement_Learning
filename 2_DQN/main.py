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

    for E in range(100):

        env.reset()
        observation = env.step(0)
        print(observation[0].shape)
        terminal = False
        #Grayscale the observation
        observation_grayscale = np.dot(observation[0][...,:3],[0.3,0.6,0.1])
        print(observation_grayscale.shape)
        observation_grayscale_resize = np.resize(observation_grayscale, (110,84))
        print(observation_grayscale_resize.shape)
        iteration = 0
        score = 0

        while not terminal :
            env.render()

            #Execute the action
            action = breakoutAgent.act(observation_grayscale_resize)
            next_state, reward, terminal, info = env.step(action)

            #Preprocess the next Observation
            next_state_grayscale = np.dot(next_state[0][...,:3],[0.3,0.6,0.1])
            next_state_grayscale_resize = np.resize(next_state_grayscale, (110,84))

            #Record the step
            breakoutAgent.record(observation_grayscale_resize, action, reward, terminal, next_state_grayscale_resize)

            #Fit the model
            if len(breakoutAgent.game_memory) >= breakoutAgent.batch_size :
                if breakoutAgent.total_iteration % breakoutAgent.train_interval == 0 :
                    breakoutAgent.fit()

            if breakoutAgent.total_iteration % breakoutAgent.update_interval == 0 :
                breakoutAgent.update_target_model()

            breakoutAgent.total_iteration += 1
            iteration += 1
            score += reward

            if terminal :
                break

        print("Episode {}/{} | score : {} | iteration : {}".format(E, 100, score, iteration))




if __name__ == "__main__":
    main()
