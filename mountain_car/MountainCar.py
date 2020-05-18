# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 12:21:23 2020
@author: Sila
"""

import numpy as np
import gym
import matplotlib.pyplot as plt

"""The original MountainCar problem was described as follows:

State variables:
(Two-dimensional continuous state space)
Velocity  = ( − 0.07 , 0.07 ) 
Position = ( − 1.2 , 0.6 ) 

Actions:
motor  = ( left , neutral , right ) 

Reward (For every time step)
reward = − 1 

For every time step, Choose an action.
Action = [ − 1 , 0 , 1 ]

Then:
Velocity = Velocity + Action*0.001 + cos(3*Position)*(-0.0025)
Position = Position + Velocity 

Starting condition:
Position  = − 0.5 
Velocity = 0

End the simulation when:
Position >= 0.6. """

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()

def MountainEnv():
    print("Mountain Environment:")
    print("env.min_position:" + str(env.min_position))
    print("env.max_position:" + str(env.max_position))
    print("env.max_speed:" + str(env.max_speed))
    print("env.goal_position:" + str(env.goal_position))
    print("env.goal_velocity:" + str(env.goal_velocity))

    print("env.observation_space.low:"+ str(env.observation_space.low))
    print("env.observation_space.high:"+ str(env.observation_space.high))

    print("Actions - env.action_space.n:"+ str(env.action_space.n))
    return

def PlayTheGame():
    done= False
    steps=0
    state = env.reset()

    while (done != True or steps<1000):
       steps= steps +1

       action = np.random.randint(0, env.action_space.n)
       #action = 2

       # Get next state and reward
       state2, reward, done, info = env.step(action)
       print("steps:"+ str(steps)+ " state:"+str(state2)+" Reward:"+str(reward)+" Done:" + str(done) + " Info:" + str(info))
       env.render()

    env.close()
    exit()

# Setup discrete
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

def Initialize_QTable():
    # Determine size of discretized state space
    q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
    return q_table

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))  # we use this tuple to look up the 3 Q values for the available actions in the q-table

# Define Q-learning function
def QLearning(env, learning, discount, epsilon, episodes):

    Q = Initialize_QTable()

    # Initialize variables to track rewards
    reward_list = []
    ave_reward_list = []

    # Calculate episodic reduction in epsilon
    # reduction = epsilon / episodes
    reduction = 0.8 / 4000

    # Run Q learning algorithm
    for episode in range(episodes):
        discrete_state = get_discrete_state(env.reset())
        done = False

        total_reward, reward = 0, 0
        while not done:

            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(Q[discrete_state])
            else:
                # Get random action
                action = np.random.randint(0, env.action_space.n)

            new_state, reward, done, _ = env.step(action)

            new_discrete_state = get_discrete_state(new_state)

            # Render environment for last five episodes
            if episode >= (episodes - 5):
                env.render()

            # new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            # If simulation did not end yet after last step - update Q table
            if not done:

                # Maximum possible Q value in next step (for new state)
                max_future_q = np.max(Q[new_discrete_state])

                # Current Q value (for current state and performed action)
                current_q = Q[discrete_state + (action,)]

                # And here's our equation for a new Q value for current state and action
                new_q = (1 - learning) * current_q + learning * (reward + discount * max_future_q)

                # Update Q table with new Q value
                Q[discrete_state + (action,)] = new_q


            # Simulation ended (for any reson) - if goal position is achived - update Q value with reward directly
            elif new_state[0] >= env.goal_position:
                # q_table[discrete_state + (action,)] = reward
                Q[discrete_state + (action,)] = 0

            discrete_state = new_discrete_state

            # Update variables
            total_reward += reward

        # Decay epsilon
        if epsilon > 0:
            epsilon -= reduction

        # Track rewards
        reward_list.append(total_reward)

        if (episode + 1) % 100 == 0:
            ave_reward = np.mean(reward_list)
            ave_reward_list.append(ave_reward)
            reward_list = []

        if (episode + 1) % 100 == 0:
            print('Episode {} Average Reward: {}'.format(episode + 1, ave_reward))

    env.close()

    return ave_reward_list

MountainEnv()
#PlayTheGame()

# Run Q-learning algorithm
rewards = QLearning(env, 0.2, 0.9, 0.8, 4000)

# Plot Rewards
plt.plot(100 * (np.arange(len(rewards)) + 1), rewards)
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Episodes')
plt.show()
