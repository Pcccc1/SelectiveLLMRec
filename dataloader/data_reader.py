import os
from collections import defaultdict
import json


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


class YelpItemProfileReader:
    def __init__(self, json_path):
        self.json_path = json_path  # dataset folder path


    def parse_profile(self, raw_str):
        raw_str = raw_str.strip()
        parts = [p.strip() for p in raw_str.split(";") if p.strip()]

        if len(parts) == 0:
            return {"name": "", "categories": []}

        name = parts[0].strip('"')
        categories = parts[1:]

        return {
            "name": name,
            "categories": categories
        }
    

    def load(self, parser):
        with open(self.json_path, "r") as f:
            raw_profiles = json.load(f)
        item_profiles = {}
        for raw_id, raw_txt in raw_profiles.items():
            if raw_id not in parser.item2id.keys():
                continue
            item_profiles[parser.item2id[raw_id]] = self.parse_profile(raw_txt)
        return item_profiles
    
    
class AmazonBookItemProfileReader:
    def __init__(self, json_path):
        self.json_path = json_path

    def parse_categories(self, cats):

        flat = set()
        for path in cats:
            for c in path:
                c = c.strip()
                if c:
                    flat.add(c)
        return list(flat)

    def load(self, parser):

        with open(self.json_path, "r") as f:
            raw = json.load(f)

        item_profiles = {}

        for raw_item_id, data in raw.items():
            if raw_item_id not in parser.item2id:
                continue

            new_id = parser.item2id[raw_item_id]

            title = data.get("title", "").strip()

            categories = data.get("categories", [])
            categories = self.parse_categories(categories)

            description = data.get("description", "").strip()

            item_profiles[new_id] = {
                "title": title,
                "categories": categories,
                "description": description
            }

        return item_profiles


class MovieItemProfileReader:

    def __init__(self, json_path):
        self.json_path = json_path

    def parse_genres(self, genre_str):
        if not genre_str:
            return []
        return [g.strip() for g in genre_str.split("|") if g.strip()]

    def load(self, parser):

        with open(self.json_path, "r") as f:
            raw = json.load(f)

        item_profiles = {}

        for raw_item_id, data in raw.items():
            # 如果 item 不在 graph dataset 里，则跳过
            if raw_item_id not in parser.item2id:
                continue

            new_id = parser.item2id[raw_item_id]

            title = data.get("title", "").strip()
            genres = self.parse_genres(data.get("genres", ""))

            item_profiles[new_id] = {
                "title": title,
                "genres": genres
            }

        return item_profiles
