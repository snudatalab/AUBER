"""
AUBER: Automated BERT-Regularization

Authors:
- Hyun Dong Lee (hl2787@columbia.edu)
- Seongmin Lee (ligi214@snu.ac.kr)
- U Kang (ukang@snu.ac.kr)
- Data Mining Lab at Seoul National University.

File: src/lib/memory.py
 - Contains source code for memory schemes.

Citation for class ReplayMemory
: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

from collections import namedtuple
import random

# memory class for storing past transitions
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.transition = namedtuple('Transition',
                ('state', 'action', 'next_state', 'reward', 'done'))

    # pushes a new transition into the memory
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
