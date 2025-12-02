from torch.utils.data import Dataset
import random

class GraphPretrainDataset(Dataset):
    def __init__(self, user_pos_items, num_users, num_items):
        self.user_pos_items = user_pos_items
        self.users = list(user_pos_items.keys())
        self.num_users = num_users
        self.num_items = num_items

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]
        pos_item = random.choice(self.user_pos_items[u])
        return u, pos_item




class GraphDatasetParser:
    #TODO: 增强的metedata还没有做映射
    def __init__(self, train, val, test):
        self.train = train
        self.val = val
        self.test = test

    def remap_ids(self):
        # 收集所有 ID
        users = set()
        items = set()
        for u, i in self.train + self.val + self.test:
            users.add(u)
            items.add(i)

        # 连续映射
        self.user2id = {u: idx for idx, u in enumerate(sorted(users))}
        self.item2id = {i: idx for idx, i in enumerate(sorted(items))}

        # 映射后的数据
        def map_one(data):
            return [(self.user2id[u], self.item2id[i]) for u, i in data]

        self.train = map_one(self.train)
        self.val = map_one(self.val)
        self.test = map_one(self.test)

        self.num_users = len(users)
        self.num_items = len(items)


    def build_user_pos_items(self):
        self.user_pos_items = {u:[] for u in range(self.num_users)}
        for u, i in self.train:
            self.user_pos_items[u].append(i)