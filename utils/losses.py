import torch
import torch.nn.functional as F


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


def semantic_alignment_loss(
    gnn_emb: torch.Tensor,
    semantic_proj_emb: torch.Tensor,
    selected_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Align projected semantic embedding with collaborative embedding
    only on selected items.
    """
    mask = selected_mask.view(-1)
    denom = mask.sum()
    if denom.item() <= 0:
        return torch.zeros((), device=gnn_emb.device)
    cos_dist = 1.0 - F.cosine_similarity(gnn_emb, semantic_proj_emb, dim=-1)
    return (cos_dist * mask).sum() / (denom + 1e-8)


def embedding_consistency_loss(
    fused_emb: torch.Tensor,
    base_emb: torch.Tensor,
    selected_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Keep fused embedding close to base GNN embedding on selected nodes
    to enforce conservative updates.
    """
    mask = selected_mask.view(-1)
    denom = mask.sum()
    if denom.item() <= 0:
        return torch.zeros((), device=fused_emb.device)
    sq = (fused_emb - base_emb).pow(2).sum(dim=-1)
    return (sq * mask).sum() / (denom + 1e-8)


def fusion_gate_l2_loss(
    gate_alpha: torch.Tensor,
    selected_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Keep fusion gate conservative on selected nodes by penalizing large gate values.
    """
    mask = selected_mask.view(-1)
    denom = mask.sum()
    if denom.item() <= 0:
        return torch.zeros((), device=gate_alpha.device)
    sq = gate_alpha.view(-1).pow(2)
    return (sq * mask).sum() / (denom + 1e-8)
