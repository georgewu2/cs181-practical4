import numpy.random as npr
import numpy as np
import csv
import sys

from SwingyMonkey import SwingyMonkey

class Learner:

    def __init__(self):
        self.last_state  = None
        self.pixelsize   = 20
        self.screen_height = 400
        self.last_action = None
        self.last_reward = None
        self.learning_rate = 1
        self.decay_rate = 1
        self.total = 0
        self.Q = dict()

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.total = 0

    def convert_to_q(self, state):
        m_pos = np.floor(state['monkey']['top']/self.pixelsize)
        t_pos = np.floor(state['tree']['top']/self.pixelsize)
        return ((self.screen_height/self.pixelsize)*(m_pos - t_pos), state['tree']['dist'])

    def update_Q(self, s, a):
        if self.last_state == None:
            return
        prev_state_index = self.convert_to_q(self.last_state)
        current_state_index = self.convert_to_q(s)
        if current_state_index in self.Q:
            q_value = max(self.Q[current_state_index][0], self.Q[current_state_index][1])
        else:   
            q_value = 0
        if prev_state_index in self.Q:
            self.Q[prev_state_index][a] += self.learning_rate * (self.last_reward + self.decay_rate * q_value - self.Q[prev_state_index][a])
        else:
            self.Q[prev_state_index] = [0,0]
            self.Q[prev_state_index][a] = self.learning_rate * (self.last_reward + self.decay_rate * q_value - self.Q[prev_state_index][a])
    def action_callback(self, state):
        '''Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.'''

        # You might do some learning here based on the current state and the last state.

        # You'll need to take an action, too, and return it.
        # Return 0 to swing and 1 to jump.
       
        self.update_Q(state, self.last_action)

        index = self.convert_to_q(state)
        if index in self.Q and 0 in self.Q[index] and 1 in self.Q[index]:
            if self.Q[index][0] > self.Q[index][1]:
                new_action = 0
            elif self.Q[index][0] < self.Q[index][1]:
                new_action = 1
        else:
            new_action = npr.rand() < 0.1
        new_state  = state

        self.last_action = new_action
        self.last_state  = new_state

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        self.total = self.total + reward
        if self.last_reward == 0:
            return 1
        self.last_reward = reward


    def total_reward(self):
        return self.total

iters = 100
learner = Learner()
with open('score_history.csv', 'w') as soln_fh:
    soln_csv = csv.writer(soln_fh,delimiter=' ',quotechar='"',quoting=csv.QUOTE_MINIMAL)

    for ii in xrange(iters):

        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,            # Don't play sounds.
                             text="Epoch %d" % (ii), # Display the epoch on screen.
                             tick_length=1,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save state
        soln_csv.writerow([learner.total_reward()])

        # Reset the state of the learner.
        learner.reset()


    
