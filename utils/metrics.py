import numpy as np

def recall_at_k(rank_list, ground_truth, k):
    """
    rank_list: 排序后的 item list，例如 [3,10,7, ...]
    ground_truth: 多正样本，例如 [10, 15, 22]
    """
    hit = len(set(rank_list[:k]) & ground_truth)
    return hit / len(ground_truth)


def ndcg_at_k(rank_list, ground_truth, k):
    rank_list = rank_list[:k]
    dcg = 0.0

    for idx, item in enumerate(rank_list):
        if item in ground_truth:
            dcg += 1.0 / np.log2(idx + 2)

    # IDCG 是正样本按最优排序取前 k 个
    ideal_hits = min(len(ground_truth), k)
    idcg = sum([1.0 / np.log2(i + 2) for i in range(ideal_hits)])

    return dcg / idcg if idcg > 0 else 0.0

def evaluate_all_ranking(
    model,
    users,
    train_user_items,
    eval_user_items,
    K=[20],
    device="cpu",
):

    model.eval()
    user_g, item_g = model.propagate()   # [U,d], [I,d]
    users = users.to(device)

    user_emb = user_g[users]             # [B,d]
    item_emb = item_g.to(device)         # [I,d]

    # 得分矩阵 [B, I]
    scores = user_emb @ item_emb.T

    recalls = {k: [] for k in K}
    ndcgs = {k: [] for k in K}

    # 用 CPU 方便处理
    scores_np = scores.detach().cpu().numpy()
    users_np = users.cpu().tolist()

    for idx, u in enumerate(users_np):
        user_score = scores_np[idx]

        # -------------------------
        # Mask 训练集中的 items
        # -------------------------
        if u in train_user_items:
            seen = train_user_items[u]
            user_score[list(seen)] = -1e9

        # 全排序
        rank_list = np.argsort(-user_score)  # 降序排序

        gt = eval_user_items[u]              # set 多正样本

        for k in K:
            recalls[k].append(recall_at_k(rank_list, gt, k))
            ndcgs[k].append(ndcg_at_k(rank_list, gt, k))

    # 最后对所有 user 求平均
    recall_res = {k: float(np.mean(recalls[k])) for k in K}
    ndcg_res   = {k: float(np.mean(ndcgs[k]))   for k in K}

    return recall_res, ndcg_res


def get_user_item_dict(data):
    user_item_dict = dict()
    for u, i in data:
        if u not in user_item_dict:
            user_item_dict[u] = set()
        user_item_dict[u].add(i)
    return user_item_dict