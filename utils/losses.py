import torch
import torch.nn.functional as F


def bpr_loss(z_user, z_pos, z_neg, reg: float = 0.0):
    pos_scores = (z_user * z_pos).sum(dim=-1)
    neg_scores = (z_user * z_neg).sum(dim=-1)
    loss = -F.logsigmoid(pos_scores - neg_scores).mean()

    if reg > 0:
        reg_term = (z_user.norm(2).pow(2) + z_pos.norm(2).pow(2) + z_neg.norm(2).pow(2)) / (
            z_user.size(0)
        )
        loss = loss + reg * reg_term
    return loss


def info_nce(anchor, positive, temperature: float = 0.2):
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    logits = anchor @ positive.t() / temperature
    labels = torch.arange(anchor.size(0), device=anchor.device)
    return F.cross_entropy(logits, labels)
