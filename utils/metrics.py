import numpy as np
import torch


def _get_idcg(length: int) -> float:
    return sum(1.0 / np.log2(i + 2) for i in range(length))


def _get_label(test_data, pred) -> np.ndarray:
    labels = []
    for gt, items in zip(test_data, pred):
        gt_set = set(gt)
        labels.append([1.0 if i in gt_set else 0.0 for i in items])
    return np.array(labels, dtype=np.float32)


def _recall_at_k_for_r(test_data, r, k: int) -> float:
    r_k = r[:, :k]
    right_pred = r_k.sum(axis=1)
    gt_len = np.array([len(gt) for gt in test_data], dtype=np.float32)
    return float(np.mean(right_pred / gt_len))


def _ndcg_at_k_for_r(test_data, r, k: int) -> float:
    r_k = r[:, :k]
    dcg = (r_k / np.log2(np.arange(2, k + 2))).sum(axis=1)
    idcg = np.array([_get_idcg(min(len(gt), k)) for gt in test_data], dtype=np.float32)
    idcg[idcg == 0] = 1.0
    return float(np.mean(dcg / idcg))


def recall_at_k(rank_list, ground_truth, k):
    """
    rank_list: 排序后的 item list，例如 [3,10,7, ...]
    ground_truth: 多正样本，例如 [10, 15, 22]
    """
    hit = len(set(rank_list[:k]) & set(ground_truth))
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


@torch.no_grad()
def evaluate_all_ranking(
    model,
    users,
    train_user_items,
    eval_user_items,
    K=[20],
    device="cuda",
    batch_size: int = 1024,
):
    model.eval()
    max_K = max(K)

    user_g, item_g = model.propagate()

    """
    userg + user_emb
    itemg + item_emb
    """

    # user_id = model.user_embedding.weight
    # item_id = model.item_embedding.weight
    # user_g = user_g + user_id
    # item_g = item_g + item_id


    users = users.to(device)
    user_g = user_g.to(device)
    item_g = item_g.to(device)

    all_ground_truth = []
    all_pred_items = []

    # 官方 LightGCN 评估流程：按用户 batch 计算评分，mask 训练正样本，再取 topK
    for start in range(0, users.size(0), batch_size):
        batch_users = users[start : start + batch_size]
        batch_emb = user_g[batch_users]                 # [B, d]
        rating = torch.matmul(batch_emb, item_g.T)      # [B, I]

        batch_user_list = batch_users.cpu().tolist()
        rating = rating.clone()
        for row, u in enumerate(batch_user_list):
            train_pos = train_user_items.get(u)
            if train_pos:
                rating[row, list(train_pos)] = float("-inf")

        _, topk_items = torch.topk(rating, k=max_K)
        topk_items = topk_items.cpu().numpy()

        for row, u in enumerate(batch_user_list):
            gt = eval_user_items.get(u)
            if not gt:
                continue
            all_ground_truth.append(list(gt))
            all_pred_items.append(topk_items[row])

    if len(all_ground_truth) == 0:
        zeros = {k: 0.0 for k in K}
        return zeros, zeros

    label_mat = _get_label(all_ground_truth, all_pred_items)
    recall_res = {k: _recall_at_k_for_r(all_ground_truth, label_mat, k) for k in K}
    ndcg_res = {k: _ndcg_at_k_for_r(all_ground_truth, label_mat, k) for k in K}

    return recall_res, ndcg_res


def get_user_item_dict(data):
    user_item_dict = dict()
    for u, i in data:
        if u not in user_item_dict:
            user_item_dict[u] = set()
        user_item_dict[u].add(i)
    return user_item_dict


@torch.no_grad()
def full_sort_scores_chunked(user_emb, item_emb, chunk_size=4096):
    """
    对 user_emb 计算其对所有 item 的打分，采用 chunk 分块的方式避免显存爆炸。
    
    输入:
        user_emb: [B, d]
        item_emb: [I, d]
        chunk_size: 每次处理多少 item（根据显存调整）
    输出:
        scores: [B, I] （在 CPU 上）
    """
    user_emb = user_emb.detach()
    item_emb = item_emb.detach()

    B = user_emb.shape[0]
    I = item_emb.shape[0]

    scores_list = []

    # 分块处理 item
    for start in range(0, I, chunk_size):
        end = min(start + chunk_size, I)

        item_chunk = item_emb[start:end]        # [C, d]
        part_scores = user_emb @ item_chunk.t() # [B, C]

        scores_list.append(part_scores.cpu())   # 放到 CPU，避免堆显存

    # 拼接成完整 [B, I]
    return torch.cat(scores_list, dim=1).numpy()
