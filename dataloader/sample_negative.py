import numpy as np

class NegativeSampler:
    def __init__(self, num_items, user_pos_items):
        self.num_items = num_items
        self.user_pos_items = user_pos_items

        # 统计 popularity
        item_pop = np.zeros(num_items)
        for items in user_pos_items.values():
            for i in items:
                item_pop[i] += 1

        # 归一化 + alias method
        self.prob = (item_pop + 1e-6) / (item_pop.sum() + 1e-6)
        self.alias = AliasMethod(self.prob)

    def sample(self, u):
        while True:
            neg = self.alias.draw()
            if neg not in self.user_pos_items[u]:
                return neg



class AliasMethod:
    def __init__(self, probs):
        self.n = len(probs)
        self.prob = np.zeros(self.n)
        self.alias = np.zeros(self.n, dtype=np.int64)

        # normalize
        probs = np.array(probs)
        probs = probs / probs.sum()

        small = []
        large = []

        scaled_probs = probs * self.n

        for i, p in enumerate(scaled_probs):
            if p < 1.0:
                small.append(i)
            else:
                large.append(i)

        while small and large:
            s = small.pop()
            l = large.pop()

            self.prob[s] = scaled_probs[s]
            self.alias[s] = l

            scaled_probs[l] = scaled_probs[l] - (1.0 - scaled_probs[s])
            if scaled_probs[l] < 1.0:
                small.append(l)
            else:
                large.append(l)

        for i in large + small:
            self.prob[i] = 1
            self.alias[i] = i

    def draw(self):
        i = np.random.randint(0, self.n)
        if np.random.rand() < self.prob[i]:
            return i
        else:
            return self.alias[i]
