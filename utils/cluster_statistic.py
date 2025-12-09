from collections import Counter
import numpy as np

class ClusterProfile:

    def __init__(self, parser, cluster_users, item_profiles):
        """
        parser: GraphDatasetParser
        cluster_users: cluster_id -> list of users
        item_profiles: new_item_id -> unified profile dict (GeneralItemProfileManager output)
        """
        self.parser = parser
        self.cluster_users = cluster_users
        self.item_profiles = item_profiles


    def compute_high_freq_items(self, top_k=20):
        cluster_top_items = {}

        for cluster_id, users in self.cluster_users.items():
            items = []
            for u in users:
                items.extend(self.parser.user_pos_items[u])

            # Count frequency
            item_counts = Counter(items)
            top_items = item_counts.most_common(top_k)

            cluster_top_items[cluster_id] = top_items

        return cluster_top_items


    def compute_cluster_statistics(self):
        cluster_behavior_stats = {}

        for cluster_id, users in self.cluster_users.items():
            lengths = [len(self.parser.user_pos_items[u]) for u in users]

            cluster_behavior_stats[cluster_id] = {
                "num_users": len(users),
                "avg_interactions": sum(lengths) / len(lengths) if lengths else 0.0,
                "median_interactions": float(np.median(lengths)) if lengths else 0.0,
                "max_interactions": max(lengths) if lengths else 0,
            }

        return cluster_behavior_stats


    def enrich_item_profiles(self, top_items):
        """
        输入: top_items = [(item_id, freq), ...]
        输出: 带完整 item profile 的列表
        """
        result = []

        for item_id, freq in top_items:
            profile = self.item_profiles.get(item_id, None)

            entry = {
                "item_id": item_id,
                "freq": freq,
                "profile": profile
            }
            result.append(entry)

        return result


    def get_cluster_profiles(self, top_k=20):
        high_freq_items = self.compute_high_freq_items(top_k)
        cluster_stats = self.compute_cluster_statistics()

        cluster_profiles = {}

        for cluster_id in self.cluster_users.keys():
            top_items = high_freq_items[cluster_id]

            # 加入 item textual profiles (统一格式)
            top_item_profiles = self.enrich_item_profiles(top_items)

            cluster_profiles[cluster_id] = {
                "statistics": cluster_stats[cluster_id],
                "top_items": top_items,
                "top_item_profiles": top_item_profiles
            }

        return cluster_profiles
