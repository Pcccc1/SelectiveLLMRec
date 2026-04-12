from .data_reader import (
    YelpItemProfileReader,
    AmazonBookItemProfileReader,
    MovieItemProfileReader,
    PickleItemProfileReader,
)

class GeneralItemProfileManager:
    """
    统一管理不同数据集的 item profiles。
    输入：dataset_name（"yelp" / "amazon" / "steam" / "movie"）
    输出：统一的 item_profiles[new_item_id]
    """
    def __init__(self, dataset_name, parser, profile_path):
        self.dataset_name = dataset_name.lower()
        self.parser = parser
        self.profile_path = profile_path

    def load(self, format="unified"):
        dataset_name = self._canonical_dataset_name(self.dataset_name)
        output_format = self._canonical_dataset_name(format)

        if self.profile_path and str(self.profile_path).lower().endswith(".pkl"):
            reader = PickleItemProfileReader(self.profile_path)
            raw_profiles = reader.load(self.parser)
            # RLMRec item profiles are plain text-like fields, keep as text-centric schema.
            return {item_id: self.to_text_profile_format(prof) for item_id, prof in raw_profiles.items()}

        if dataset_name == "yelp":
            reader = YelpItemProfileReader(self.profile_path)
        elif dataset_name == "amazon":
            reader = AmazonBookItemProfileReader(self.profile_path)
        elif dataset_name == "movie":
            reader = MovieItemProfileReader(self.profile_path)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        raw_profiles = reader.load(self.parser)
        unified = {}

        if output_format == "yelp" and dataset_name == "yelp":
            for item_id, prof in raw_profiles.items():
                unified[item_id] = self.to_yelp_format(prof)
            return unified
        elif output_format == "amazon" and dataset_name == "amazon":
            for item_id, prof in raw_profiles.items():
                unified[item_id] = self.to_amazon_format(prof)
            return unified
        elif output_format == "movie" and dataset_name == "movie":
            for item_id, prof in raw_profiles.items():
                unified[item_id] = self.to_movie_format(prof)
            return unified
        else:
            for item_id, prof in raw_profiles.items():
                unified[item_id] = self.to_unified_format(prof, dataset_name)

        return unified

    def _canonical_dataset_name(self, name):
        name = str(name).lower()
        aliases = {
            "amazon_book": "amazon",
            "amazon-book": "amazon",
        }
        return aliases.get(name, name)

    def to_unified_format(self, prof, dataset):
        """
        将各数据集的结构转换为统一 schema。
        """
        title = None
        name = None
        categories = []
        genres = []
        description = None

        if dataset == "yelp":
            name = prof.get("name", "")
            categories = prof.get("categories", [])

        elif dataset == "amazon":
            title = prof.get("title", "")
            categories = prof.get("categories", [])
            description = prof.get("description", "")

        elif dataset == "movie":
            title = prof.get("title", "")
            genres = prof.get("genres", [])
        elif dataset == "steam":
            description = prof.get("profile", "")

        return {
            "title": title,
            "name": name,
            "categories": categories,
            "genres": genres,
            "description": description,
            "raw": prof
        }

    def to_text_profile_format(self, prof):
        if isinstance(prof, dict):
            profile_text = prof.get("profile")
            return {"profile": profile_text if isinstance(profile_text, str) else str(profile_text)}
        if isinstance(prof, str):
            return {"profile": prof}
        return {"profile": str(prof)}

    def to_yelp_format(self, prof):
        name = prof.get("name", "")
        categories = prof.get("categories", [])

        return {
            "name": name,
            "categories": categories
        }

    def to_amazon_format(self, prof):
        title = prof.get("title", "")
        categories = prof.get("categories", [])
        description = prof.get("description", "")

        return {
            "title": title,
            "categories": categories,
            "description": description
        }
    
    def to_movie_format(self, prof):
        title = prof.get("title", "")
        genres = prof.get("genres", [])

        return {
            "title": title,
            "genres": genres
        }   
