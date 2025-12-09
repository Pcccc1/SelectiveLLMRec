from .data_reader import YelpItemProfileReader, AmazonBookItemProfileReader, MovieItemProfileReader

class GeneralItemProfileManager:
    """
    统一管理不同数据集的 item profiles。
    输入：dataset_name（"yelp" / "amazon" / "movie"）
    输出：统一的 item_profiles[new_item_id]
    """
    def __init__(self, dataset_name, parser, profile_path):
        self.dataset_name = dataset_name.lower()
        self.parser = parser
        self.profile_path = profile_path

    def load(self, format="unified"):
        if self.dataset_name == "yelp":
            reader = YelpItemProfileReader(self.profile_path)
        elif self.dataset_name == "amazon_book":
            reader = AmazonBookItemProfileReader(self.profile_path)
        elif self.dataset_name == "movie":
            reader = MovieItemProfileReader(self.profile_path)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        raw_profiles = reader.load(self.parser)
        unified = {}

        if format == "yelp" and self.dataset_name == "yelp":
            for item_id, prof in raw_profiles.items():
                unified[item_id] = self.to_yelp_format(prof)
            return unified
        elif format == "amazon" and self.dataset_name == "amazon_book":
            for item_id, prof in raw_profiles.items():
                unified[item_id] = self.to_amazon_format(prof)
            return unified
        elif format == "movie" and self.dataset_name == "movie":
            for item_id, prof in raw_profiles.items():
                unified[item_id] = self.to_movie_format(prof)
            return unified
        else:
            for item_id, prof in raw_profiles.items():
                unified[item_id] = self.to_unified_format(prof, self.dataset_name)

        return unified

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

        return {
            "title": title,
            "name": name,
            "categories": categories,
            "genres": genres,
            "description": description,
            "raw": prof
        }

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