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


def info_nce(anchor, positive, temperature: float = 0.2):
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    logits = anchor @ positive.t() / temperature
    labels = torch.arange(anchor.size(0), device=anchor.device)
    return F.cross_entropy(logits, labels)
