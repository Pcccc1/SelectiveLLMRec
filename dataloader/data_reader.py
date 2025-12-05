import os
from collections import defaultdict


class DataReader:
    def __init__(self, path, min_user_interactions: int = 0, min_item_interactions: int = 0):
        self.path = path   # dataset folder path
        self.min_user_interactions = max(0, min_user_interactions)
        self.min_item_interactions = max(0, min_item_interactions)

    def read_interactions(self, filename):
        filepath = os.path.join(self.path, filename)
        data = []
        with open(filepath, "r") as f:
            for line in f:
                parts = line.strip().split()
                u = parts[0]
                i = parts[1]
                data.append((u, i))
        return data

    def load_all(self):
        train = self.read_interactions("train.txt")
        val = self.read_interactions("val.txt")
        test = self.read_interactions("test.txt")

        if self.min_user_interactions > 0 or self.min_item_interactions > 0:
            train, val, test = self._filter_min_interactions(train, val, test)

        return train, val, test

    def _filter_min_interactions(self, train, val, test):
        """
        Iteratively filter out users/items that do not meet min interaction thresholds
        across all splits, matching the usual LightGCN preprocessing.
        """
        combined = [("train", *x) for x in train] + [("val", *x) for x in val] + [("test", *x) for x in test]

        while True:
            user_cnt = defaultdict(int)
            item_cnt = defaultdict(int)
            for _, u, i in combined:
                user_cnt[u] += 1
                item_cnt[i] += 1

            filtered = []
            for split, u, i in combined:
                if user_cnt[u] >= self.min_user_interactions and item_cnt[i] >= self.min_item_interactions:
                    filtered.append((split, u, i))

            if len(filtered) == len(combined):
                break
            combined = filtered

        train_f = [(u, i) for split, u, i in combined if split == "train"]
        val_f = [(u, i) for split, u, i in combined if split == "val"]
        test_f = [(u, i) for split, u, i in combined if split == "test"]
        return train_f, val_f, test_f
