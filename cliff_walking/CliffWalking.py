# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 09:12:53 2017
"""

# Eaaa project. October 2017. Exam project files 2020.
# Reinforcement Learning.
# Classic Cliff Walking, as stated by Barto and Sutton.

import numpy as np
from collections import deque
import matplotlib.pyplot as plt

ROWS = 4
COLS = 12
S = (3, 0)
G = (3, 11)

class Cliff:

    def __init__(self):
        self.end = False
        self.pos = S
        self.board = np.zeros([4, 12])
        # add cliff marked as -1
        self.board[3, 1:11] = -1

    def nxtPosition(self, action):
        if action == "up":
            nxtPos = (self.pos[0] - 1, self.pos[1])
        elif action == "down":
            nxtPos = (self.pos[0] + 1, self.pos[1])
        elif action == "left":
            nxtPos = (self.pos[0], self.pos[1] - 1)
        else:
            nxtPos = (self.pos[0], self.pos[1] + 1)
        # check legitimacy
        if nxtPos[0] >= 0 and nxtPos[0] <= 3:
            if nxtPos[1] >= 0 and nxtPos[1] <= 11:
                self.pos = nxtPos

        if self.pos == G:
            self.end = True
            print("Game ends reaching goal")
        if self.board[self.pos] == -1:
            self.end = True
            print("Game ends falling off cliff")

        return self.pos

    def giveReward(self):
        # give reward
        if self.pos == G:
            return -1
        if self.board[self.pos] == 0:
            return -1
        return -100

    def show(self):
        for i in range(0, ROWS):
            print('-------------------------------------------------')
            out = '| '
            for j in range(0, COLS):
                if self.board[i, j] == -1:
                    token = '*'
                if self.board[i, j] == 0:
                    token = '0'
                if (i, j) == self.pos:
                    token = 'S'
                if (i, j) == G:
                    token = 'G'
                out += token + ' | '
            print(out)
        print('-------------------------------------------------')

def showRoute(states):
    board = np.zeros([4, 12])
    # add cliff marked as -1
    board[3, 1:11] = -1
    for i in range(0, ROWS):
        print('-------------------------------------------------')
        out = '| '
        for j in range(0, COLS):
            token = '0'
            if board[i, j] == -1:
                token = '*'
            if (i, j) in states:
                token = 'R'
            if (i, j) == G:
                token = 'G'
            out += token + ' | '
        print(out)
    print('-------------------------------------------------')

class Agent:

    def __init__(self, exp_rate=0.3, lr=0.1, sarsa=True):
        self.cliff = Cliff()
        self.actions = ["up", "left", "right", "down"]
        self.states = []  # record position and action of each episode
        self.pos = S
        self.exp_rate = exp_rate
        self.lr = lr
        self.sarsa = sarsa
        self.state_actions = {}
        self.avg_window = 50
        self.samp_rewards = deque(maxlen=self.avg_window)
        self.avg_rewards = deque(maxlen=0)

        for i in range(ROWS):
            for j in range(COLS):
                self.state_actions[(i, j)] = {}
                for a in self.actions:
                    self.state_actions[(i, j)][a] = 0

    def chooseAction(self):
        # epsilon-greedy
        mx_nxt_reward = -999
        action = ""

        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # greedy action
            for a in self.actions:
                current_position = self.pos
                nxt_reward = self.state_actions[current_position][a]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
        return action

    def reset(self):
        self.states = []
        self.cliff = Cliff()
        self.pos = S

    def play(self, rounds=10):
        self.avg_rewards = deque(maxlen=rounds)
        for _ in range(rounds):
            while 1:
                curr_state = self.pos
                cur_reward = self.cliff.giveReward()
                action = self.chooseAction()

                # next position
                self.cliff.pos = self.cliff.nxtPosition(action)
                self.pos = self.cliff.pos

                self.states.append([curr_state, action, cur_reward])
                if self.cliff.end:
                    break
            # game end update estimates
            reward = self.cliff.giveReward()

            # collecting data for graphing
            self.samp_rewards.append(reward)

            print("End game reward", reward)
            # reward of all actions in end state is same
            for a in self.actions:
                self.state_actions[self.pos][a] = reward

            # in sarsa reward is based on the taken action, not the best action.
            if self.sarsa:
                for s in reversed(self.states):
                    pos, action, r = s[0], s[1], s[2]
                    current_value = self.state_actions[pos][action]
                    reward = current_value + self.lr * (r + reward - current_value)
                    self.state_actions[pos][action] = round(reward, 3) # 3 decimals
            else:
                for s in reversed(self.states):
                    pos, action, r = s[0], s[1], s[2]
                    current_value = self.state_actions[pos][action]
                    reward = current_value + self.lr * (r + reward - current_value)
                    self.state_actions[pos][action] = round(reward, 3) # 3 decimals
                    # update using the max value of S'
                    reward = np.max(list(self.state_actions[pos].values()))  # max

            # Experimenting with decay, nothing conclusive so far
            # self.lr = self.lr*0.995
            # collecting data for graphing
            self.avg_rewards.append(np.average(self.samp_rewards))
            self.reset()

# Show the cliff world
print("Cliff World:")
c = Cliff()
c.show()

sarsa = False
qlearning= True

ag_learner = Agent(exp_rate=0.02, lr=1, sarsa=False)
ag_learner.play(rounds=80) # We start with 500 rounds of learning. See question a.

# Q-learning - What did we find.
ag_optimal = Agent(exp_rate=0) # set exploration_rate to 0. See question c
ag_optimal.state_actions = ag_learner.state_actions

print()
print("Q-learning route")
states = []

while 1:
    curr_state = ag_optimal.pos
    if states.count(curr_state) > 5:
        break
    action = ag_optimal.chooseAction()
    states.append(curr_state)
    print("current position {} |action {}".format(curr_state, action))

    # next position
    ag_optimal.cliff.pos = ag_optimal.cliff.nxtPosition(action)
    ag_optimal.pos = ag_optimal.cliff.pos

    if ag_optimal.cliff.end:
        break

print()
print("Q-learning route:")
showRoute(states)

#plotting the average reward for Q-Learning avg_window
# plt.plot(ag_learner.avg_rewards)
# plt.title('Q-Learning reward ' + repr(ag_learner.avg_window) + ' round aveage development')
# plt.ylabel('Reward')
# plt.xlabel('Round')
# plt.show()
#

sarsa = True
qlearning= False

# Calculate using Sarsa
#exp rate for SARSA kan decide how "afraid" the final ageint is of the cliff.
ag_sarsa = Agent(exp_rate=0.1, lr=0.1, sarsa=True)
ag_sarsa.play(rounds=3000)

# Sarsa
ag_op = Agent(exp_rate=0)
ag_op.state_actions = ag_sarsa.state_actions

states = []
while 1:
    curr_state = ag_op.pos
    action = ag_op.chooseAction()
    states.append(curr_state)
    print("current position {} |action {}".format(curr_state, action))

    # next position
    ag_op.cliff.pos = ag_op.cliff.nxtPosition(action)
    ag_op.pos = ag_op.cliff.pos

    if ag_op.cliff.end:
        break

print()
print("Sarsa route:")
showRoute(states)

#plotting the average reward for SARSA avg_window
plt.plot(ag_sarsa.avg_rewards)
plt.title('SARSA ' + repr(ag_learner.avg_window) + ' round aveage development')
plt.ylabel('Reward')
plt.xlabel('Round')
plt.show()

# # Display both graphs
# plt.plot(ag_learner.avg_rewards)
# plt.plot(ag_sarsa.avg_rewards)
# plt.show()