import numpy.random as npr
import sys

from SwingyMonkey import SwingyMonkey

class Learner:

    def __init__(self):
        self.last_state  = None
        self.pixelsize   = 10
        self.screen_height = 400
        self.last_action = None
        self.last_reward = None
        self.learning_rate = None
        self.decay_rate = None
        self.Q = 

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def convert_to_q(state):
        m_pos = ceil(state['monkey']['top']/self.pixelsize)
        t_pos = ceil(state['tree']['top']/self.pixelsize)

        return ((self.screen_height/self.pixelsize)*m_pos + t_pos)

    def update_Q(s, a):
        prev_state_index = convert_to_Q(self.last_state)
        current_state_index = convert_to_Q(s)
        Q(prev_state_index, a) += self.learning_rate * (self.last_reward + self.decay_rate * (max(Q(current_state_index,0), Q(current_state_index,1))) - Q(prev_state_index,a))

    def action_callback(self, state):
        '''Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.'''

        # You might do some learning here based on the current state and the last state.

        # You'll need to take an action, too, and return it.
        # Return 0 to swing and 1 to jump.



        new_action = npr.rand() < 0.1
        new_state  = state

        self.last_action = new_action
        self.last_state  = new_state

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward

iters = 100
learner = Learner()

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

    # Reset the state of the learner.
    learner.reset()



    
