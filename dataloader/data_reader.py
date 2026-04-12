import os
from collections import defaultdict
import json
import pickle
from pathlib import Path


class _CompatUnpickler(pickle.Unpickler):
    """
    Handle backward/forward module path changes in scipy/numpy pickles.
    RLMRec sparse matrices are sometimes pickled with scipy>=1.8 internals
    (e.g. scipy.sparse._coo) which are unavailable in older scipy versions.
    """

    MODULE_ALIASES = {
        "scipy.sparse._coo": "scipy.sparse.coo",
        "scipy.sparse._csr": "scipy.sparse.csr",
        "scipy.sparse._csc": "scipy.sparse.csc",
        "scipy.sparse._dia": "scipy.sparse.dia",
        "scipy.sparse._dok": "scipy.sparse.dok",
        "scipy.sparse._lil": "scipy.sparse.lil",
        "scipy.sparse._bsr": "scipy.sparse.bsr",
        "scipy.sparse._base": "scipy.sparse.base",
        "scipy.sparse._data": "scipy.sparse.data",
        "scipy.sparse._sputils": "scipy.sparse.sputils",
        "numpy._core": "numpy.core",
        "numpy._core.multiarray": "numpy.core.multiarray",
    }

    def find_class(self, module, name):
        return super().find_class(self.MODULE_ALIASES.get(module, module), name)


def load_pickle_compat(path: str):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError:
            f.seek(0)
            return _CompatUnpickler(f).load()


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
        root = Path(self.path)
        split_paths = {
            "train": root / "trn_mat.pkl",
            "val": root / "val_mat.pkl",
            "test": root / "tst_mat.pkl",
        }

        if all(p.exists() for p in split_paths.values()):
            train = self._read_sparse_interactions(split_paths["train"])
            val = self._read_sparse_interactions(split_paths["val"])
            test = self._read_sparse_interactions(split_paths["test"])
        else:
            legacy_paths = [root / "train.txt", root / "val.txt", root / "test.txt"]
            if all(p.exists() for p in legacy_paths):
                raise RuntimeError(
                    f"Legacy dataset format is deprecated at {root}. "
                    "Please switch to RLMRec files: trn_mat.pkl / val_mat.pkl / tst_mat.pkl."
                )
            missing = ", ".join(str(p) for p in split_paths.values() if not p.exists())
            raise FileNotFoundError(f"Missing RLMRec split files: {missing}")

        if self.min_user_interactions > 0 or self.min_item_interactions > 0:
            train, val, test = self._filter_min_interactions(train, val, test)

        return train, val, test

    def _read_sparse_interactions(self, file_path: Path):
        mat = load_pickle_compat(str(file_path))

        if hasattr(mat, "tocoo"):
            coo = mat.tocoo()
            pairs = [
                (int(u), int(i))
                for u, i, v in zip(coo.row, coo.col, coo.data)
                if float(v) > 0.0
            ]
            return pairs

        # Dense fallback
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("numpy is required to parse dense interaction matrices.") from exc

        arr = np.asarray(mat)
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2D matrix in {file_path}, got shape={arr.shape}")
        users, items = np.nonzero(arr > 0)
        return [(int(u), int(i)) for u, i in zip(users.tolist(), items.tolist())]

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


class PickleItemProfileReader:
    def __init__(self, pkl_path):
        self.pkl_path = pkl_path

    def _normalize_profile(self, raw_profile):
        if isinstance(raw_profile, dict):
            if "profile" in raw_profile:
                profile_text = raw_profile.get("profile")
                return {"profile": profile_text if isinstance(profile_text, str) else str(profile_text)}
            return raw_profile
        if isinstance(raw_profile, str):
            return {"profile": raw_profile}
        return {"profile": str(raw_profile)}

    def load(self, parser):
        raw_profiles = load_pickle_compat(self.pkl_path)
        if not isinstance(raw_profiles, dict):
            raise ValueError(f"Expected dict in {self.pkl_path}, got {type(raw_profiles)}")

        item_profiles = {}
        for raw_id, raw_profile in raw_profiles.items():
            if raw_id not in parser.item2id:
                continue
            item_profiles[parser.item2id[raw_id]] = self._normalize_profile(raw_profile)
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
