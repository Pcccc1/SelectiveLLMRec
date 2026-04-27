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


def _minmax_norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min() + eps)


class NodeValueEvaluator:

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
        v_gp = self.get_collaborative_structure_gap(self.item_id_emb, self.item_emb_layers[-1])
        
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
        v_pr_item_norm = _minmax_norm(v_pr_item)
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
        v_unc_norm = _minmax_norm(v_unc)
        return v_unc_norm


    def get_collaborative_structure_gap(self, item_id_emb: torch.Tensor, item_g: torch.Tensor):
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
        
        v_gap_norm = _minmax_norm(v_gap)

        return v_gap_norm

    # Backward-compatible alias with old typo.
    def get_collaborativ_structure_gap(self, item_id_emb: torch.Tensor, item_g: torch.Tensor):
        return self.get_collaborative_structure_gap(item_id_emb, item_g)


# Backward-compatible alias kept for legacy imports.
Node_value_Evaluator = NodeValueEvaluator


class UserNodeValueEvaluator:
    """
    Estimate per-user semantic need score with graph-value signals:
      - PageRank exposure
      - layer-wise uncertainty
      - ID/GNN representation gap
      - low-activity boost
    """

    def __init__(
        self,
        parser,
        user_emb_layers: list[torch.Tensor],
        user_id_emb: torch.Tensor,
        alpha: float = 0.33,
        beta: float = 0.33,
        gamma: float = 0.34,
        cold_start_boost: float = 0.2,
    ):
        self.parser = parser
        self.user_emb_layers = user_emb_layers
        self.user_id_emb = user_id_emb
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.cold_start_boost = float(cold_start_boost)

    def calculate(self) -> torch.Tensor:
        v_pr = self.get_pagerank_users()
        v_un = self.get_uncertainty(self.user_emb_layers)
        v_gp = self.get_collaborative_structure_gap(self.user_id_emb, self.user_emb_layers[-1])
        low_activity = 1.0 - self.get_user_activity()

        score = (
            self.alpha * v_pr
            + self.beta * v_un
            + self.gamma * v_gp
            + self.cold_start_boost * low_activity
        )
        return _minmax_norm(score)

    def get_pagerank_users(
        self,
        alpha: float = 0.15,
        max_iter: int = 100,
        tol: float = 1e-8,
    ) -> torch.Tensor:
        adj = _convert_sp_mat_to_sp_tensor(self.parser.adj_mat)
        N = adj.size(0)
        device = adj.device

        degree = torch.sparse.sum(adj, dim=1).to_dense()
        dangling = degree == 0

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

        v_pr_user = pr[: self.parser.num_users]
        return _minmax_norm(v_pr_user)

    def get_uncertainty(self, user_emb_layers: list[torch.Tensor]) -> torch.Tensor:
        assert len(user_emb_layers) >= 2, "Need at least 2 layers to compute uncertainty"
        num_layers = len(user_emb_layers) - 1
        device = user_emb_layers[0].device
        v_unc = torch.zeros(user_emb_layers[0].size(0), device=device)

        for l in range(1, num_layers + 1):
            diff = user_emb_layers[l] - user_emb_layers[l - 1]
            v_unc += torch.norm(diff, dim=1)

        v_unc = v_unc / num_layers
        return _minmax_norm(v_unc)

    def get_collaborative_structure_gap(
        self, user_id_emb: torch.Tensor, user_g: torch.Tensor
    ) -> torch.Tensor:
        assert user_id_emb.shape == user_g.shape
        v_gap = torch.norm(user_id_emb - user_g, dim=1)
        return _minmax_norm(v_gap)

    # Backward-compatible alias with old typo.
    def get_collaborativ_structure_gap(
        self, user_id_emb: torch.Tensor, user_g: torch.Tensor
    ) -> torch.Tensor:
        return self.get_collaborative_structure_gap(user_id_emb, user_g)

    def get_user_activity(self) -> torch.Tensor:
        act = torch.zeros(self.parser.num_users, dtype=torch.float32)
        for u, _ in self.parser.train:
            act[u] += 1.0
        return _minmax_norm(act)
