from collections import namedtuple, deque
import random

# Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'candidates', 'reward'))
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'chosen', 'qid'))

class Memory(object):
    '''Replay memory to store experiences'''

    def __init__(self, capacity) -> None:
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        '''Save a transition in memory buffer'''
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)