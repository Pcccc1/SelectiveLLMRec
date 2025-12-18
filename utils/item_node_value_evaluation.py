import torch


def _convert_sp_mat_to_sp_tensor(sp_mat) -> torch.sparse.FloatTensor:
    """scipy.spmatrix -> torch.sparse.FloatTensor (coalesced)."""
    coo = sp_mat.tocoo()
    indices = torch.stack(
        (torch.from_numpy(coo.row), torch.from_numpy(coo.col)), dim=0
    )
    values = torch.from_numpy(coo.data.astype("float32"))
    shape = coo.shape
    sp_tensor = torch.sparse_coo_tensor(indices, values, torch.Size(shape))
    return sp_tensor.coalesce()


class Node_value_Evaluator:

    def __init__(self, parser, item_emb_layers: list, item_id_emb: torch.Tensor, alpha: float = 0.33, beta: float = 0.33, gamma: float = 0.34):
        self.parser = parser
        self.item_emb_layers = item_emb_layers
        self.item_id_emb = item_id_emb
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def calculate(self):
        v_pr = self.get_pagerank()
        v_un = self.get_uncertainty(self.item_emb_layers)
        v_gp = self.get_collaborativ_structure_gap(self.item_id_emb, self.item_emb_layers[-1])
        
        v = (
            self.alpha * v_pr +
            self.beta * v_un +
            self.gamma * v_gp
        )
        return v


    def get_pagerank(
        self,   
        alpha: float = 0.15,
        max_iter: int = 100,
        tol: float = 1e-8,
    ):
        
        adj = _convert_sp_mat_to_sp_tensor(self.parser.adj_mat)
        N = adj.size(0)
        device = adj.device

        # degree
        degree = torch.sparse.sum(adj, dim=1).to_dense()
        dangling = degree == 0

        # build P = D^{-1} A
        deg_inv = torch.zeros_like(degree)
        deg_inv[degree > 0] = 1.0 / degree[degree > 0]

        idx = adj.indices()
        val = adj.values() * deg_inv[idx[0]]
        P = torch.sparse_coo_tensor(idx, val, adj.size()).coalesce()

        pr = torch.full((N,), 1.0 / N, device=device)
        teleport = pr.clone()

        for _ in range(max_iter):
            pr_next = (1 - alpha) * torch.sparse.mm(
                P.transpose(0, 1), pr.unsqueeze(1)
            ).squeeze(1)

            dangling_mass = pr[dangling].sum()
            pr_next += (1 - alpha) * dangling_mass * teleport
            pr_next += alpha * teleport

            if torch.norm(pr_next - pr, p=1) < tol:
                pr = pr_next
                break

            pr = pr_next

        v_pr_item = pr[self.parser.num_users:]
        minmax_norm = lambda x, eps=1e-12: (x - x.min()) / (x.max() - x.min() + eps)
        v_pr_item_norm = minmax_norm(v_pr_item)
        return v_pr_item_norm


    def get_uncertainty(self, item_emb_layers: list):

        assert len(item_emb_layers) >= 2, "Need at least 2 layers to compute uncertainty"

        num_layers = len(item_emb_layers) - 1
        device = item_emb_layers[0].device

        v_unc = torch.zeros(
            item_emb_layers[0].size(0),
            device=device
        )

        for l in range(1, num_layers + 1):
            diff = item_emb_layers[l] - item_emb_layers[l - 1]   # [I, d]
            v_unc += torch.norm(diff, dim=1)                      # L2 per item

        v_unc = v_unc / num_layers
        minmax_norm = lambda x, eps=1e-12: (x - x.min()) / (x.max() - x.min() + eps)
        v_unc_norm = minmax_norm(v_unc)
        return v_unc_norm


    def get_collaborativ_structure_gap(self, item_id_emb: torch.Tensor, item_g: torch.Tensor):
        """
        Semantic gap between ID embedding and final GNN embedding.

        Args:
            item_id_emb: Tensor [num_items, dim]
            item_g: Tensor [num_items, dim]

        Returns:
            v_gap: Tensor [num_items]
        """
        assert item_id_emb.shape == item_g.shape

        # L2 distance per item
        v_gap = torch.norm(item_id_emb - item_g, dim=1)
        
        minmax_norm = lambda x, eps=1e-12: (x - x.min()) / (x.max() - x.min() + eps)
        v_gap_norm = minmax_norm(v_gap)

        return v_gap_norm
