"""
Q Learning 

Author: Shota Horii <sh.sinker@gmail.com>
"""

import numpy as np

class QLearning:
    def __init__(self, n_states, n_actions, lr=0.5, discount_factor=0.99):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.discount_factor = discount_factor
        self.q_table = np.random.uniform(low=-1, high=1, size=(n_states, n_actions))
    
    def action(self, state, episode):
        """
        get action to take, based on the current state and current episode.

        Parameters
        ----------
        state: int
        current state encoded as an int (=row index of the q table.)
        
        episode: int
        current episode

        Returns
        -------
        next_action: int
        """
        # epsilon-greedy algorithm: take more "adventurous" actions earlier
        # and take optimal actions later. 
        eps = 0.5 * (1 / (episode + 1)) 
        if eps <= np.random.uniform(0, 1):
            next_action = np.argmax(self.q_table[state])
        else:
            next_action = np.random.choice(range(self.n_actions))
        return next_action

    def update(self, prev_state, action_taken, reward, state):
        gamma = self.discount_factor
        alpha = self.lr
        max_q = self.q_table[state].max()
        self.q_table[prev_state, action_taken] = (1 - alpha) * self.q_table[prev_state, action_taken] + alpha * (reward + gamma * max_q)