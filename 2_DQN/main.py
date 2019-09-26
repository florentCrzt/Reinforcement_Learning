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
        terminal = False
        #Grayscale the observation
        observation_grayscale = np.dot(observation[0][...,:3],[0.3,0.6,0.1])
        observation_grayscale_resize = np.resize(observation_grayscale, (110,84))
        iteration = 1
        buffer = []
        score = 0
        buffer.append(observation_grayscale_resize)
        while not terminal :
            env.render()

            #Execute .act every 4 steps, the remains do 0
            if iteration%4 == 0 :
                buffer = np.transpose(buffer, (1,2,0))
                buffer = np.expand_dims(buffer, axis=0)
                action = breakoutAgent.act(buffer)
                next_state, reward, terminal, info = env.step(action)
                buffer = []
            else :
                action = 2
                next_state, reward, terminal, info = env.step(action)

            #Preprocess the next Observation
            next_state_grayscale = np.dot(next_state[0][...,:3],[0.3,0.6,0.1])
            next_state_grayscale_resize = np.resize(next_state_grayscale, (110,84))
            buffer.append(next_state_grayscale_resize)

            #Record the step
            breakoutAgent.record(observation_grayscale_resize, action, reward, next_state_grayscale_resize,terminal)

            #Fit the model
            if len(breakoutAgent.game_memory) >= breakoutAgent.batch_size :
                if breakoutAgent.total_iteration % breakoutAgent.train_interval == 0 :
                    breakoutAgent.fit()

            if breakoutAgent.total_iteration % breakoutAgent.update_interval == 0 :
                breakoutAgent.update_target_model()

            breakoutAgent.total_iteration += 1
            observation_grayscale_resize = next_state_grayscale_resize
            score += reward
            iteration += 1
            if terminal :
                break

        print("Episode {}/{} | score : {} | explo : {}".format(E, 100, score, breakoutAgent.exploration_rate))




if __name__ == "__main__":
    main()
