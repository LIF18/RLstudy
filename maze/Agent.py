import numpy as np

from random import choice
import random

class Agent:
    ### START CODE HERE ###

    def __init__(self, actions):
        self.actions = actions
        self.epsilon = 1

    def choose_action(self, observation):
        action = np.random.choice(self.actions)
        return action


class Dyna_Q_Agent:
    def __init__(self, actions, alpha, gamma):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.q_dic = {}
        self.state_list = []
        self.action_dic = {}
        self.method_dic = {}

        self.epsilon = 0.1
        self.actions = actions
        self.t = 0

    def greedy(self,state):
        state = tuple(state)
        if state not in self.q_dic:
            self.q_dic[state] = {}
            tmp = self.q_dic[state]
            for a in self.actions:
                if a not in tmp:
                    tmp[a] = 0

        max_action = self.max_q(state)
        action = choice(max_action)
        return action
        self.t += 1
        if self.t % 1000 == 0:
            self.epsilon /= 2
        ran = random.random()
        if ran < self.epsilon:
            action = choice(self.actions)
        else:
            action = choice(max_action)
        return action


    def max_q(self,state):
        value = []
        tmp = self.q_dic[state]
        for a in self.actions:
            value.append(tmp[a])

        max_q = max(value)
        actions = []
        for action, value in enumerate(value):
            if max_q == value:
                actions.append(action)
        return actions

    def update(self,s,s_,a,r):
        s = tuple(s)

        s_ = tuple(s_)
        tmp = []

        if s_ not in self.q_dic:
            self.q_dic[s_] = {}
            t = self.q_dic[s_]
            for action in self.actions:
                if action not in t:
                    t[action] = 0

        t = self.q_dic[s_]
        for action in self.actions:
            tmp.append(t[action])

        max_q1 = max(tmp)

        self.q_dic[s][a] = (1 - self.alpha)*self.q_dic[s][a] + self.alpha*(r + self.gamma*max_q1)


    def train(self,s,s_,a,r):
        if s not in self.state_list:
            self.state_list.append(s)
        s = tuple(s)
        if s not in self.action_dic:
            self.action_dic[s] = [a]
        elif a not in self.action_dic[s]:
            self.action_dic[s].append(a)

        method = (s,a)
        self.method_dic[method] = (s_,r)

    def model(self,n = 50):
        for i in range(n):
            s = choice(self.state_list)
            a = choice(self.action_dic[tuple(s)])
            method = (tuple(s), a)
            s_, r = self.method_dic[method]

            #self.update(s,s_,a,r)
            self.update(s,s_,a,r)
    ### END CODE HERE ###
