import os

class DataReader:
    def __init__(self, path):
        self.path = path   # dataset folder path

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
        return train, val, test
