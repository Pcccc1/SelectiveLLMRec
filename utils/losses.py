import torch
import torch.nn.functional as F
from torch import nn


def bpr_loss(z_user, z_pos, z_neg, reg: float = 0.0, user_id_emb=None, pos_id_emb=None, neg_id_emb=None):
    pos_scores = (z_user * z_pos).sum(dim=-1)
    neg_scores = (z_user * z_neg).sum(dim=-1)
    loss = -F.logsigmoid(pos_scores - neg_scores).mean()

    if reg > 0 and user_id_emb is not None and pos_id_emb is not None and neg_id_emb is not None:
        reg_term = (
            user_id_emb.norm(2).pow(2)
            + pos_id_emb.norm(2).pow(2)
            + neg_id_emb.norm(2).pow(2)
        ) / user_id_emb.size(0)
        loss = loss + 0.5 * reg * reg_term
    return loss


def info_nce(anchor, positive, temperature: float = 0.2):
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    logits = anchor @ positive.t() / temperature
    labels = torch.arange(anchor.size(0), device=anchor.device)
    return F.cross_entropy(logits, labels)


def cluster_info_nce(
    id_emb: torch.Tensor,        # [B, d]
    llm_emb: torch.Tensor,       # [B, d]
    cluster_id: torch.Tensor,    # [B]
    temperature: float = 0.1,
):
    """
    InfoNCE with cluster-based positives.
    LLM embedding is used as supervision signal (no gradient).
    """

    # -------- 1. normalize --------
    z_id = F.normalize(id_emb, dim=-1)                  # [B, d]
    z_llm = F.normalize(llm_emb.detach(), dim=-1)       # [B, d]

    # -------- 2. similarity --------
    # sim[i, j] = cos(id_i, llm_j)
    sim = torch.matmul(z_id, z_llm.t()) / temperature   # [B, B]

    # -------- 3. positive mask (same cluster) --------
    # pos_mask[i, j] = 1 if cluster_i == cluster_j
    cluster_i = cluster_id.view(-1, 1)                  # [B, 1]
    cluster_j = cluster_id.view(1, -1)                  # [1, B]
    pos_mask = (cluster_i == cluster_j).float()         # [B, B]

    # 防止自己和自己 trivially 对齐（可选，但建议）
    self_mask = torch.eye(pos_mask.size(0), device=pos_mask.device)
    pos_mask = pos_mask * (1.0 - self_mask)

    # -------- 4. log-softmax --------
    log_prob = F.log_softmax(sim, dim=1)                 # [B, B]

    # -------- 5. InfoNCE loss --------
    # 对每个 i，只在正样本上求期望
    pos_count = pos_mask.sum(dim=1)                      # [B]

    # 避免某些 batch 中“孤簇”导致 NaN
    valid = pos_count > 0

    loss = -(log_prob * pos_mask).sum(dim=1) / (pos_count + 1e-8)
    loss = loss[valid].mean()

    return loss

