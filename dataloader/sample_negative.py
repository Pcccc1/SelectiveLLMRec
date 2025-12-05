import numpy as np

class NegativeSampler:
    def __init__(self, num_items, user_pos_items):
        self.num_items = num_items
        self.user_pos_items = user_pos_items

    def sample(self, u):
        while True:
            neg = np.random.randint(0, self.num_items)
            if neg not in self.user_pos_items[u]:
                return neg
