# -*- coding:utf-8 -*-
# DQN homework.
import os
import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

from gym import wrappers
from utils import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# hyper-parameter.  
EPISODES = 4000

class DQNAgent:
    def __init__(self, state_size, action_size):
        # if you want to see MsPacman learning, then change to True
        self.render = False

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        # These are hyper parameters for the DQN
        self.discount_factor = 0.3#gamma
        self.learning_rate = 0.005
        self.epsilon = 0.2
        self.epsilon_min = 0.01
        #self.epsilon_decay = (self.epsilon-self.epsilon_min) / 10
        self.epsilon_decay = self.epsilon_min / 5000
        self.batch_size = 32
        self.train_start = 1000
        # create replay memory using deque
        self.maxlen = 8000
        self.memory = deque(maxlen=self.maxlen)#双向队列
        # create main model
        self.model_target = self.build_model()
        self.model_eval = self.build_model()

    # approximate Q function using Neural Network
    # you can modify the network to get higher reward.
    def build_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(32, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        #loss = mean_squard_error
        return model

    # get action from model using epsilon-greedy policy
    def get_action(self, state,episode):
        '''
        if episode <= 1000:
            self.epsilon = 1 - episode*0.0001
        elif episode <= 3000:
            self.epsilon = 0.8 - (episode - 1000) *0.0003
        elif episode <= 5000:
            self.episode = 0.2*pow(0.9985,episode-3000)
        '''
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model_eval.predict(state)
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        # epsilon decay.
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model_eval.predict(update_input)
        target_val = self.model_target.predict(update_target) #kersa predict

        for i in range(self.batch_size):
            # Q Learning: get maximum Q value at s' from model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    np.amax(target_val[i])) #r + gamma* qmax

        # and do the model fit!
        self.model_eval.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)
    
    def eval2target(self):
        self.model_target.set_weights(self.model_eval.get_weights())

if __name__ == "__main__":
    # load the gym env
    env = gym.make('MsPacman-ram-v0')
    # set  random seeds to get reproduceable result(recommended)
    set_random_seed(0)
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    # create the agent
    agent = DQNAgent(state_size, action_size)
    # log the training result
    scores, episodes = [], []
    graph_episodes = []
    graph_score = []
    avg_length = 10
    sum_score = 0
    avg_score = 0
    update = 50
    # train DQN
    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        lives = 3
        check = 0
        tmp = 0
        while not done: 
            dead = False         
            while not dead:
                # render the gym env
                if agent.render:
                    env.render()
                # get action for the current state
                action = agent.get_action(state,e)
                # take the action in the gym env, obtain the next state
                next_state, reward, done, info = env.step(action)
                next_state = np.reshape(next_state, [1, state_size])
                # judge if the agent dead
                dead = info['ale.lives']<lives
                lives = info['ale.lives']
                # update score value
                score += reward
                # save the sample <s, a, r, s'> to the replay memory
                agent.append_sample(state,action,reward,next_state,done)
                if done:
                    break
                # train the evaluation network
                agent.train_model()
                # go to the next state
                state = next_state
            # update the target network after some iterations.
            if check >= update:
                agent.eval2target()
                check = 0
            check += 1

        # print info and draw the figure.
        if done:
            scores.append(score)
            sum_score += score
            episodes.append(e)
            # plot the reward each episode
            # pylab.plot(episodes, scores, 'b')
            print("episode:", e, "  score:", score, "  memory length:",
                    len(agent.memory), "  epsilon:", agent.epsilon)
        if e%avg_length == 0:
            graph_episodes.append(e)
            graph_score.append(sum_score / avg_length)
            avg_score += sum_score / avg_length
            sum_score = 0
            # plot the reward each avg_length episodes
    pylab.plot(graph_episodes, graph_score, 'r')
    pylab.savefig("./pacman5_avg.png")
        
        # save the network if you want to test it.