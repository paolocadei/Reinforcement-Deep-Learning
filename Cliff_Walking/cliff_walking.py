#!/usr/bin/env python
# coding: utf-8

## Modified based on https://github.com/openai/gym/blob/master/gym/envs/toy_text/cliffwalking.py
## Prepared by Yi He, Feb 2022

import numpy as np
import matplotlib.pyplot as plt
import sys
from io import StringIO

###################
# cliff_walking environment
###################
class CliffWalkingEnv():
    """
    This is a simple implementation of the Gridworld Cliff
    reinforcement learning task.

    Adapted from Example 6.6 (page 106) from Reinforcement Learning: An Introduction
    by Sutton and Barto:
    http://incompleteideas.net/book/bookdraft2018jan1.pdf

    With inspiration from:
    https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/cliff_walking.py

    The board is a 4x12 matrix, with (using NumPy matrix indexing):
        [3, 0] as the start at bottom-left
        [3, 11] as the goal at bottom-right
        [3, 1..10] as the cliff at bottom-center

    Each time step incurs -1 reward, and stepping into the cliff incurs -100 reward
    and a reset to the start. An episode terminates when the agent reaches the goal.
    """

    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self):

        self.shape = (4, 12)
        self.start_state = (3,0)
        self.start_state_index = np.ravel_multi_index((3, 0), self.shape)
        self.terminal_state = (self.shape[0] - 1, self.shape[1] - 1)
        self.terminal_state_index = np.ravel_multi_index(self.terminal_state, self.shape)

        self.nS = np.prod(self.shape)

        self.action_space = ['U', 'D', 'L', 'R']
        self.nA = len(self.action_space)
        self.delta_space = [[-1, 0],[1, 0],[0, -1],[0, 1]]

        # Cliff Location
        self._cliff = np.zeros(self.shape, dtype=bool)
        self._cliff[3, 1:-1] = True

        # Calculate transition probabilities and rewards
        self.P = {}
        for s in range(self.nS):
            for a in range(self.nA):
                self.P[(s,a)] = self._calculate_transition_prob(s, a)

        # Initialize state
        self.s = self.start_state_index

    def _limit_coordinates(self, coord):
        """
        Prevent the agent from falling out of the grid world
        :param coord:
        :return:
        """
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, s, a):
        """
        Determine the outcome for an action. Transition Prob is always 1.0.
        :return: (1.0, new_state, reward, done)
        """
        new_position = np.array(np.unravel_index(s, self.shape)) + np.array(self.delta_space[a])
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        if self._cliff[tuple(new_position)]:
            return [(1.0, self.start_state_index, -100, False)]

        is_done = tuple(new_position) == self.terminal_state
        return [(1.0, new_state, -1, is_done)]

    def step(self, a):

        transitions = self.P[(self.s, a)]
        i = np.random.choice(range(len(transitions)), p=[t[0] for t in transitions])
        p, s, reward, d = transitions[i]
        self.s = s
        self.lastaction = a

        return s, reward, d

    def reset(self):

        self.s = self.start_state_index
        self.lastaction = None

        return self.s
    
###################
# policies
###################
def generate_random_policy(cw, deterministic=False, seed=None):
    """
    Randomly generates a deterministic or probabilistic policy pi(a|s).

    Returns:
        ndarray: 2-dimensional array of size (N x 4), where N is the number of states of gridworld.
        The 4 columns correspond, respectively, to the actions "U", "D", "L", "R".
    """

    if seed is not None:
        np.random.seed(seed)
    
    if deterministic:
        probs = np.zeros((cw.nS, 4))
        probs[np.arange(cw.nS), np.random.randint(4, size=(cw.nS,))] = 1.0
    else:
        logits = np.random.normal(size=(cw.nS, 4), scale=1)
        exp = np.exp(logits)
        probs = (exp.T / np.sum(exp, axis=1)).T

    # Setting the state values of the terminal states to zero
    probs[cw.terminal_state_index] = [0.0, 0.0, 0.0, 0.0]

    return(probs)

def select_action(state, Q, cw, epsilon=0.0):

    if np.random.rand() < epsilon:
        action_ = np.random.randint(cw.nA)
    else:
        action_ = np.random.choice(np.nonzero(Q[state] == np.max(Q[state]))[0])

    return action_            

###################
# simulation
###################
def simulate_episode(cw, Q, epsilon=0.1, max_iter=float('inf')):
    """
    Simulate a episode following a given policy
    @return: resulting states, actions and rewards for the entire episode
    """
    state = cw.reset()
    rewards = []
    actions = []
    states = [state]
    is_done = False
    Num_iter = 0
    
    while not is_done and Num_iter<max_iter:

        action = select_action(state, Q, cw, epsilon=epsilon)
        actions.append(action)
        state, reward, is_done = cw.step(action)
        states.append(state)
        rewards.append(reward)
        if is_done:
            break
        Num_iter += 1

    #if not is_done:
        #print("Episode did not reach terminal state.")

    return states, actions, rewards



###################
# print functions
###################
def color_special_states_func(cw, grid_):

    grid_[cw.start_state] = 0.0, 0.0, 1.0, 1.0
    grid_[cw.terminal_state] = 0.0, 1.0, 0.0, 1.0

    for s in range(cw.nS):
        position = np.unravel_index(s, cw.shape)
        if cw._cliff[position]:
            grid_[position] = 1.0, 0.0, 0.0, 1.0
    
    return grid_

def print_grid(cw, print_states=False, color_terminal_states=True, ax=None):

    # Each state is a white, transparent tile
    grid_ = np.ones((*(cw.shape), 4))
    grid_[:,:,-1] = 0.0

    # Color terminal states
    if color_terminal_states:
        grid_ = color_special_states_func(cw, grid_)

    # Plotting
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(grid_)

    # Minor ticks
    ax.set_xticks(np.arange(-.5, cw.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, cw.shape[0], 1), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)

    # Remove minor ticks
    ax.tick_params(which='minor', bottom=False, left=False)

    # Remove other ticks and labels
    ax.tick_params(which='major', bottom=False, left=False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    # Print states
    if print_states:
        for j in range(cw.nS):
            ax.text(j%cw.shape[1], j//cw.shape[1], j, ha="center", va="center")
        
def print_policy(probs, cw, ax=None, color_terminal_states=True, **kwargs):

    if ax is None:
        fig, ax = plt.subplots()
        print_grid(cw, ax=ax, color_terminal_states=color_terminal_states, **kwargs)

    # Print policy
    for j in range(cw.nS):
        for direction_idx in range(4):
            prob = probs[j, direction_idx]
            direction = [(0.0, -0.25), (0.0, 0.25), (-0.25, 0), (0.25, 0)][direction_idx] # U, D, L R
            ax.arrow(j%cw.shape[1], j//cw.shape[1], *direction, head_width=0.1, alpha=prob)