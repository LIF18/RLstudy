from maze_env import Maze
from Agent import Dyna_Q_Agent

import time


if __name__ == "__main__":
    ### START CODE HERE ###
    # This is an agent with random policy. You can learn how to interact with the environment through the code below.
    # Then you can delete it and write your own code.

    env = Maze()
    agent = Dyna_Q_Agent(actions=list(range(env.n_actions)), alpha = 0.5, gamma = 0.9)
    #agent = cmAgent1(list(range(env.n_actions)), epsilon=0.5, N=30, gamma=0.9, alpha=0.5)
    for episode in range(50):
        s = env.reset() #state
        episode_reward = 0
        while True:

            #env.render()                 # You can comment all render() to turn off the graphical interface in training process to accelerate your code.
            a = agent.greedy(s)
            s_, r, done = env.step(a)
            agent.update(s,s_,a,r)
            agent.train(s,s_,a,r)

            episode_reward += r
            s = s_
            agent.model()
            if done:
                #env.render()
                #time.sleep(0.5)
                break
        print('episode:', episode, 'episode_reward:', episode_reward)

    ### END CODE HERE ###

    print('\ntraining over\n')
