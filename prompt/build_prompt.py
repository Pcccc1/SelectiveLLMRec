def build_cluster_text(profile):
    stats = profile["statistics"]
    top_item_profiles = profile["top_item_profiles"]

    # Merge all categories for cluster-level semantic summary
    all_cats = []
    for entry in top_item_profiles:
        cats = entry["profile"].get("categories", []) or entry["profile"].get("genres", [])
        all_cats.extend(cats)

    cat_str = ", ".join(list(set(all_cats)))

    lines = [
        "Cluster Preference Summary:",
        f"User count: {stats['num_users']}",
        f"Typical categories: {cat_str}",
        "Representative items:",
    ]

    for entry in top_item_profiles[:5]:  # 只取前 5 个，提高效率
        name = entry["profile"].get("name") or entry["profile"].get("title")
        cats = entry["profile"].get("categories", []) or entry["profile"].get("genres", [])
        cat_list = ", ".join(cats)
        lines.append(f"- {name}: {cat_list}")

    return "\n".join(lines)
