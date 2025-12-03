import random
from scipy.sparse import lil_matrix
class NegativeSampler:

    def __init__(self, num_items, user_pos_items):
        self.num_items = num_items
        self.user_pos_items = user_pos_items

    def sample(self, u):
        """Sample a negative item for user u."""
        while True:
            neg = random.randint(0, self.num_items - 1)
            if neg not in self.user_pos_items[u]:
                return neg
