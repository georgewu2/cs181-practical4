import numpy.random as npr
import numpy as np
import csv
import sys
import random

from SwingyMonkey import SwingyMonkey

class Learner:

    def __init__(self):
        self.last_state  = None
        self.pixelsize   = 10
        self.screen_height = 400
        self.last_action = None
        self.last_reward = None

        # initial learning rate or alpha in Q-learning
        self.learning_rate = 0.25

        # gamma in Q-learning
        self.decay_rate = 0.9

        # total reward
        self.total = 0

        # holds Q-values. keys are a 5-tuple defined by convert_to_q,
        # and values are a list containing two entries, one for each action
        self.Q = dict()

        # dictionary of values to scale down learning rates by
        self.alphas = dict()

        # for epsilon greedy
        self.iteration = 100

    def reset(self, i):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.total = 0
        self.iteration = (1+i)


    def convert_to_q(self, state):
        """ Converts state into key for q dictionary
        Takes in a state with structure defined in SwingyMonkey.py
        Returns a 5-tuple consisting of:
        (   <1 if top of monkey is above 300 pixels>,
            <1 if bottom of monkey is below 50 pixels>,
            <difference between top of monkey and top of tree>,
            <1 if pixels to next tree trunk is less than 100>,
            <1 if velocity of monkey is positive>
            )
        """
        m_pos       = np.floor(state['monkey']['top']/self.pixelsize)
        t_pos       = np.floor(state['tree']['top']/self.pixelsize)
        m_near_top  = state['monkey']['top'] > 300
        m_near_bot  = state['monkey']['bot'] < 50
        m_close     = state['tree']['dist'] < 100
        m_vel       = state['monkey']['vel'] > 0
        return ((m_near_top, m_near_bot,m_pos-t_pos,m_close, m_vel))

    def update_Q(self, s, a):
        """ Updates the relevant values of Q
        This is called every time the state changes.
        """
        # On the first iteration we can't update Q, because we haven't transitioned to a
        # different state.
        if self.last_state == None:
            return

        # get keys of both states
        prev_state_key = self.convert_to_q(self.last_state)
        current_state_key = self.convert_to_q(s)

        # get the maximum q-values
        if current_state_key in self.Q:
            q_value = max(self.Q[current_state_key][0], self.Q[current_state_key][1])
        else:   
            q_value = 0

        # update the values
        if prev_state_key in self.Q:
            self.Q[prev_state_key][a] += (self.learning_rate/self.alphas[prev_state_key][a]) * (self.last_reward + self.decay_rate * q_value - self.Q[prev_state_key][a])
        else:
            self.Q[prev_state_key] = [0,0]
            self.Q[prev_state_key][a] = (self.learning_rate/self.alphas[prev_state_key][a]) * (self.last_reward + self.decay_rate * q_value - self.Q[prev_state_key][a])
    
    def action_callback(self, state):
        '''Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.'''

        # You might do some learning here based on the current state and the last state.

        # You'll need to take an action, too, and return it.
        # Return 0 to swing and 1 to jump.
        
        # first update the q-value
        self.update_Q(state, self.last_action)

        # decide the next action to take
        index = self.convert_to_q(state)
        if index in self.Q:
            if self.Q[index][0] > self.Q[index][1]:
                new_action = 0
            elif self.Q[index][0] < self.Q[index][1]:
                new_action = 1
            else:
                new_action = 0
        else:
            new_action = 0
        
        if random.random() < 1.0/self.iteration:
            new_action = 1 - new_action

        # update our alphas
        new_state  = state
        new_state_index = self.convert_to_q(new_state)
        if new_state_index in self.alphas:
            self.alphas[new_state_index][new_action] += 1
        else:
            self.alphas[new_state_index] = [0,0]
            self.alphas[new_state_index][new_action]  = 1

        self.last_action = new_action
        self.last_state  = new_state

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        self.total = self.total + reward
        self.last_reward = reward


    def total_reward(self):
        return self.total

    def score(self):
        return self.last_state['score']

iters = 150
learner = Learner()
with open('score_history.csv', 'wb') as soln_fh:
    soln_csv = csv.writer(soln_fh,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
    scores = []
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
        soln_csv.writerow([learner.total_reward(), learner.score()])
        scores.append(learner.total_reward())
        # Reset the state of the learner.
        learner.reset(ii)
    print np.mean(scores)
    
