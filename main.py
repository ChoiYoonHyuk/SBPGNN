import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import add_self_loops, remove_self_loops, to_undirected

EPS = 1e-12


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def accuracy(probs: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    pred = probs[mask].argmax(dim=-1)
    return (pred == y[mask]).float().mean().item()


@torch.no_grad()
def ece_score(probs: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, n_bins: int = 15) -> float:
    p = probs[mask]
    yy = y[mask]
    conf, pred = p.max(dim=-1)
    correct = (pred == yy).float()
    bins = torch.linspace(0.0, 1.0, n_bins + 1, device=p.device)
    ece = torch.zeros([], device=p.device)
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        in_bin = (conf > lo) & (conf <= hi) if b < n_bins - 1 else (conf >= lo) & (conf <= hi)
        frac = in_bin.float().mean()
        if frac.item() > 0:
            acc_bin = correct[in_bin].mean()
            conf_bin = conf[in_bin].mean()
            ece = ece + frac * (acc_bin - conf_bin).abs()
    return ece.item()


def get_split_masks(data, split_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    if train_mask.dim() == 1:
        return train_mask, val_mask, test_mask
    if split_idx < 0 or split_idx >= train_mask.size(1):
        raise ValueError(f"split_idx={split_idx} out of range for mask shape {train_mask.shape}")
    return train_mask[:, split_idx], val_mask[:, split_idx], test_mask[:, split_idx]


def compute_reverse_edge(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    src, dst = edge_index[0], edge_index[1]
    key = src * num_nodes + dst
    rev_key = dst * num_nodes + src
    sorted_key, perm = torch.sort(key)
    pos = torch.searchsorted(sorted_key, rev_key)
    pos = pos.clamp(max=sorted_key.numel() - 1)
    rev = perm[pos]
    if not torch.all(sorted_key[pos] == rev_key):
        bad = (sorted_key[pos] != rev_key).nonzero(as_tuple=False).view(-1)
        first_bad = bad[0].item()
        s = src[first_bad].item()
        d = dst[first_bad].item()
        raise RuntimeError(
            f"Reverse edge missing for edge ({s}->{d}). Make sure edge_index is undirected (both directions exist)."
        )
    return rev


def load_dataset(name: str, root: str = "./data"):
    name_raw = name
    name = name.lower()
    transform = NormalizeFeatures()

    if name in ["cora", "citeseer", "pubmed"]:
        from torch_geometric.datasets import Planetoid

        proper = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}[name]
        dataset = Planetoid(root=os.path.join(root, proper), name=proper, split="public", transform=transform)
        data = dataset[0]
        return data, dataset.num_features, dataset.num_classes

    if name in ["chameleon", "squirrel"]:
        try:
            from torch_geometric.datasets import WikipediaNetwork

            dataset = WikipediaNetwork(
                root=os.path.join(root, name_raw),
                name=name,
                geom_gcn_preprocess=True,
                transform=transform,
            )
            data = dataset[0]
            return data, dataset.num_features, dataset.num_classes
        except Exception as e:
            print(f"[WARN] WikipediaNetwork load failed ({e}). Trying fallback...")

    if name in ["texas", "wisconsin", "cornell"]:
        try:
            from torch_geometric.datasets import WebKB

            dataset = WebKB(root=os.path.join(root, name_raw), name=name, transform=transform)
            data = dataset[0]
            return data, dataset.num_features, dataset.num_classes
        except Exception as e:
            print(f"[WARN] WebKB load failed ({e}). Trying fallback...")

    try:
        from torch_geometric.datasets import HeterophilousGraphDataset

        dataset = HeterophilousGraphDataset(root=os.path.join(root, name_raw), name=name, transform=transform)
        data = dataset[0]
        return data, dataset.num_features, dataset.num_classes
    except Exception as e:
        raise RuntimeError(
            f"Could not load dataset '{name_raw}'. Planetoid/WikipediaNetwork/WebKB/HeterophilousGraphDataset failed.\nLast error: {e}"
        )


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x


class UnaryEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        encoder_type: str = "gat",
        gat_heads: int = 8,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.gat_heads = gat_heads

        if encoder_type == "gcn":
            self.conv1 = GCNConv(in_dim, hidden_dim, cached=True, normalize=True)
            self.conv2 = GCNConv(hidden_dim, num_classes, cached=True, normalize=True)
            self.res1 = nn.Linear(in_dim, hidden_dim, bias=False)
            self.norm1 = nn.LayerNorm(hidden_dim)
        elif encoder_type == "gat":
            if hidden_dim % gat_heads != 0:
                raise ValueError(f"hidden_dim must be divisible by gat_heads (hidden_dim={hidden_dim}, gat_heads={gat_heads})")
            out_per_head = hidden_dim // gat_heads
            self.conv1 = GATConv(in_dim, out_per_head, heads=gat_heads, concat=True, dropout=dropout)
            self.conv2 = GATConv(hidden_dim, num_classes, heads=1, concat=False, dropout=dropout)
            self.res1 = nn.Linear(in_dim, hidden_dim, bias=False)
            self.norm1 = nn.LayerNorm(hidden_dim)
        elif encoder_type == "mlp":
            self.lin1 = nn.Linear(in_dim, hidden_dim)
            self.lin2 = nn.Linear(hidden_dim, num_classes)
        else:
            raise ValueError(f"Unknown encoder_type={encoder_type}")

    def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.encoder_type == "gcn":
            if edge_index is None:
                raise ValueError("gcn encoder requires edge_index")
            x0 = x
            x = F.dropout(x, p=self.dropout, training=self.training)
            h = self.conv1(x, edge_index)
            h = h + self.res1(x0)
            h = self.norm1(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            logits = self.conv2(h, edge_index)
            return h, logits

        if self.encoder_type == "gat":
            if edge_index is None:
                raise ValueError("gat encoder requires edge_index")
            x0 = x
            x = F.dropout(x, p=self.dropout, training=self.training)
            h = self.conv1(x, edge_index)
            h = h + self.res1(x0)
            h = self.norm1(h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            logits = self.conv2(h, edge_index)
            return h, logits

        h = self.lin1(x)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        logits = self.lin2(h)
        return h, logits


@dataclass
class InferenceConfig:
    T: int = 10
    eta: float = 0.2
    init_unary: bool = True
    adaptive_outer: int = 1
    eta_min: float = 0.05
    eta_max: float = 0.8
    use_cert: bool = False
    cert_k: int = 10
    cert_eps_fd: float = 1e-3
    use_uncert: bool = False
    hutch_samples: int = 4
    hutch_eps_fd: float = 1e-3
    alpha_scale: float = 1.0


class CertBP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        edge_hidden_dim: int = 64,
        dropout: float = 0.6,
        w_max: float = 0.8,
        cmp_margin: float = 0.05,
        encoder_type: str = "gat",
        gat_heads: int = 8,
        alpha_max: float = 1.5,
        mix_init: float = 0.98,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.w_max = w_max
        self.cmp_margin = cmp_margin
        self.alpha_max = alpha_max

        self.encoder = UnaryEncoder(in_dim, hidden_dim, num_classes, dropout, encoder_type=encoder_type, gat_heads=gat_heads)
        self.edge_mlp = MLP(in_dim=2 * hidden_dim + 2, hidden_dim=edge_hidden_dim, out_dim=1, dropout=dropout)

        self.R_raw = nn.Parameter(torch.empty(num_classes, num_classes))
        nn.init.xavier_uniform_(self.R_raw)
        with torch.no_grad():
            self.R_raw += 0.5 * torch.eye(num_classes)

        self.R_scale_log = nn.Parameter(torch.tensor(0.0))
        self.msg_logit = nn.Parameter(torch.tensor(-3.0))

        mix_init = float(min(max(mix_init, 1e-4), 1.0 - 1e-4))
        self.mix_logit = nn.Parameter(torch.tensor(np.log(mix_init / (1.0 - mix_init)), dtype=torch.float))

        self.a_u = nn.Parameter(torch.tensor(0.01))
        self.b_u = nn.Parameter(torch.tensor(0.0))
        self.a_eta = nn.Parameter(torch.tensor(2.0))
        self.b_eta = nn.Parameter(torch.tensor(0.0))

    def _msg_alpha(self, alpha_scale: float = 1.0) -> torch.Tensor:
        return (self.alpha_max * torch.sigmoid(self.msg_logit)) * float(alpha_scale)

    def _sym_R(self) -> torch.Tensor:
        R = 0.5 * (self.R_raw + self.R_raw.t())
        scale = F.softplus(self.R_scale_log) + 1e-6
        return scale * torch.tanh(R)

    @staticmethod
    def _centered_log(m: torch.Tensor) -> torch.Tensor:
        logm = torch.log(m.clamp(min=EPS))
        return logm - logm.mean(dim=-1, keepdim=True)

    def _edge_struct_features(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        deg = torch.bincount(src, minlength=num_nodes).float()
        logdeg = torch.log(deg + 1.0)
        a = logdeg[src]
        b = logdeg[dst]
        return torch.stack([a + b, (a - b).abs()], dim=-1)

    def _build_pairwise_kernel(
        self, h: torch.Tensor, edge_index: torch.Tensor, struct_feat: torch.Tensor, rev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index[0], edge_index[1]
        edge_in = torch.cat([h[src] * h[dst], (h[src] - h[dst]).abs(), struct_feat], dim=-1)
        w_raw = self.edge_mlp(edge_in).squeeze(-1)
        w = self.w_max * torch.sigmoid(w_raw)
        w = 0.5 * (w + w[rev])
        R = self._sym_R()
        K = torch.exp(w.view(-1, 1, 1) * R.view(1, R.size(0), R.size(1)))
        return w, K

    def _edge_norm(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        deg = torch.bincount(src, minlength=num_nodes).float().clamp(min=1.0)
        return (deg[src] * deg[dst]).pow(-0.5)

    def _bp_one_run(
        self,
        log_phi: torch.Tensor,
        K: torch.Tensor,
        edge_index: torch.Tensor,
        rev: torch.Tensor,
        num_nodes: int,
        T: int,
        eta: float,
        init_unary: bool,
        alpha_scale: float,
    ) -> torch.Tensor:
        device = log_phi.device
        C = log_phi.size(1)
        src, dst = edge_index[0], edge_index[1]
        alpha = self._msg_alpha(alpha_scale)
        edge_norm = self._edge_norm(edge_index, num_nodes).to(device)

        if init_unary:
            m = torch.softmax(log_phi[src], dim=-1)
        else:
            m = torch.full((edge_index.size(1), C), 1.0 / C, device=device)

        for _ in range(T):
            f = torch.bmm(m.unsqueeze(1), K).squeeze(1)
            log_f = torch.log(f.clamp(min=EPS))
            log_f = log_f * edge_norm.view(-1, 1)

            sum_in = torch.zeros((num_nodes, C), device=device)
            sum_in.index_add_(0, dst, log_f)

            excl = sum_in[src] - log_f[rev]
            log_msg = log_phi[src] + alpha * excl
            m_new = torch.softmax(log_msg, dim=-1)

            m = (1.0 - eta) * m + eta * m_new
            m = m.clamp(min=EPS)
            m = m / m.sum(dim=-1, keepdim=True)

        return m

    def _compute_beliefs(
        self,
        log_phi: torch.Tensor,
        K: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        m: torch.Tensor,
        alpha_scale: float,
    ) -> torch.Tensor:
        device = log_phi.device
        C = log_phi.size(1)
        dst = edge_index[1]
        alpha = self._msg_alpha(alpha_scale)
        edge_norm = self._edge_norm(edge_index, num_nodes).to(device)

        f = torch.bmm(m.unsqueeze(1), K).squeeze(1)
        log_f = torch.log(f.clamp(min=EPS))
        log_f = log_f * edge_norm.view(-1, 1)

        sum_in = torch.zeros((num_nodes, C), device=device)
        sum_in.index_add_(0, dst, log_f)

        log_b = log_phi + alpha * sum_in
        return torch.softmax(log_b, dim=-1)

    def _F_undamped_map(
        self,
        u: torch.Tensor,
        log_phi: torch.Tensor,
        K: torch.Tensor,
        edge_index: torch.Tensor,
        rev: torch.Tensor,
        num_nodes: int,
        alpha_scale: float,
    ) -> torch.Tensor:
        device = u.device
        src, dst = edge_index[0], edge_index[1]
        C = u.size(1)
        alpha = self._msg_alpha(alpha_scale)
        edge_norm = self._edge_norm(edge_index, num_nodes).to(device)

        m = torch.softmax(u, dim=-1)
        f = torch.bmm(m.unsqueeze(1), K).squeeze(1)
        log_f = torch.log(f.clamp(min=EPS))
        log_f = log_f * edge_norm.view(-1, 1)

        sum_in = torch.zeros((num_nodes, C), device=device)
        sum_in.index_add_(0, dst, log_f)

        excl = sum_in[src] - log_f[rev]
        log_msg = log_phi[src] + alpha * excl
        m_new = torch.softmax(log_msg, dim=-1)
        return self._centered_log(m_new)

    @torch.no_grad()
    def _matvec_Jv_fd(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        log_phi: torch.Tensor,
        K: torch.Tensor,
        edge_index: torch.Tensor,
        rev: torch.Tensor,
        num_nodes: int,
        eps_fd: float,
        alpha_scale: float,
    ) -> torch.Tensor:
        Fu = self._F_undamped_map(u, log_phi, K, edge_index, rev, num_nodes, alpha_scale)
        Fu2 = self._F_undamped_map(u + eps_fd * v, log_phi, K, edge_index, rev, num_nodes, alpha_scale)
        return (Fu2 - Fu) / eps_fd

    def _matvec_JtJv(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        log_phi: torch.Tensor,
        K: torch.Tensor,
        edge_index: torch.Tensor,
        rev: torch.Tensor,
        num_nodes: int,
        eps_fd: float,
        alpha_scale: float,
    ) -> torch.Tensor:
        u_var = u.detach().requires_grad_(True)
        Fu = self._F_undamped_map(u_var, log_phi, K, edge_index, rev, num_nodes, alpha_scale)

        with torch.no_grad():
            Jv = self._matvec_Jv_fd(u.detach(), v.detach(), log_phi, K, edge_index, rev, num_nodes, eps_fd, alpha_scale)
            y = Jv.detach()

        scalar = (Fu * y).sum()
        (grad_u,) = torch.autograd.grad(scalar, u_var, retain_graph=False, create_graph=False)
        return grad_u.detach()

    @torch.no_grad()
    def estimate_delta_power(
        self,
        u_star: torch.Tensor,
        log_phi: torch.Tensor,
        K: torch.Tensor,
        edge_index: torch.Tensor,
        rev: torch.Tensor,
        num_nodes: int,
        k: int,
        eps_fd: float,
        alpha_scale: float,
    ) -> float:
        v = torch.randn_like(u_star)
        v = v / (v.norm() + EPS)

        for _ in range(k):
            Av = self._matvec_JtJv(u_star, v, log_phi, K, edge_index, rev, num_nodes, eps_fd, alpha_scale)
            n = Av.norm() + EPS
            v = Av / n

        Av = self._matvec_JtJv(u_star, v, log_phi, K, edge_index, rev, num_nodes, eps_fd, alpha_scale)
        lam = (v * Av).sum().item() / ((v * v).sum().item() + EPS)
        return float(1.0 - lam)

    @torch.no_grad()
    def hutchinson_diag_JJt(
        self,
        u_star: torch.Tensor,
        log_phi: torch.Tensor,
        K: torch.Tensor,
        edge_index: torch.Tensor,
        rev: torch.Tensor,
        num_nodes: int,
        samples: int,
        eps_fd: float,
        alpha_scale: float,
    ) -> torch.Tensor:
        Fu = self._F_undamped_map(u_star, log_phi, K, edge_index, rev, num_nodes, alpha_scale)
        acc = torch.zeros_like(Fu)

        for _ in range(samples):
            z = torch.empty_like(u_star).bernoulli_(0.5)
            z = z * 2.0 - 1.0
            Fuz = self._F_undamped_map(u_star + eps_fd * z, log_phi, K, edge_index, rev, num_nodes, alpha_scale)
            Jz = (Fuz - Fu) / eps_fd
            acc += Jz * Jz

        return acc / float(samples)

    def compatibility_prior(self) -> torch.Tensor:
        R = self._sym_R()
        diag = torch.diag(R).view(-1, 1)
        margin = self.cmp_margin
        M = R - diag + margin
        mask = 1.0 - torch.eye(R.size(0), device=R.device)
        return F.relu(M) * mask

    def forward(self, data, inf_cfg: InferenceConfig) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = data.x
        edge_index = data.edge_index
        rev = data.rev_edge
        num_nodes = data.num_nodes

        h, logits = self.encoder(x, edge_index)
        log_phi = logits

        struct = self._edge_struct_features(edge_index, num_nodes)
        _, K = self._build_pairwise_kernel(h, edge_index, struct, rev)

        eta_used = inf_cfg.eta
        m = None

        for _ in range(max(1, inf_cfg.adaptive_outer)):
            m = self._bp_one_run(
                log_phi=log_phi,
                K=K,
                edge_index=edge_index,
                rev=rev,
                num_nodes=num_nodes,
                T=inf_cfg.T,
                eta=eta_used,
                init_unary=inf_cfg.init_unary,
                alpha_scale=inf_cfg.alpha_scale,
            )

            if inf_cfg.adaptive_outer > 1 and inf_cfg.use_cert:
                u_star = self._centered_log(m).detach()
                delta = self.estimate_delta_power(
                    u_star=u_star,
                    log_phi=log_phi.detach(),
                    K=K.detach(),
                    edge_index=edge_index,
                    rev=rev,
                    num_nodes=num_nodes,
                    k=inf_cfg.cert_k,
                    eps_fd=inf_cfg.cert_eps_fd,
                    alpha_scale=inf_cfg.alpha_scale,
                )
                sig = torch.sigmoid(self.a_eta * torch.tensor(delta, device=x.device) + self.b_eta).item()
                eta_used = inf_cfg.eta_min + (inf_cfg.eta_max - inf_cfg.eta_min) * sig

        b = self._compute_beliefs(log_phi, K, edge_index, num_nodes, m, alpha_scale=inf_cfg.alpha_scale)

        extras: Dict[str, torch.Tensor] = {
            "beliefs": b,
            "messages": m,
            "eta_used": torch.tensor(eta_used, device=x.device),
            "alpha": self._msg_alpha(inf_cfg.alpha_scale).detach(),
            "mix": torch.sigmoid(self.mix_logit).detach(),
        }

        probs_bp = b

        if inf_cfg.use_uncert:
            u_star = self._centered_log(m).detach()
            diag = self.hutchinson_diag_JJt(
                u_star=u_star,
                log_phi=log_phi.detach(),
                K=K.detach(),
                edge_index=edge_index,
                rev=rev,
                num_nodes=num_nodes,
                samples=inf_cfg.hutch_samples,
                eps_fd=inf_cfg.hutch_eps_fd,
                alpha_scale=inf_cfg.alpha_scale,
            )
            src = edge_index[0]
            node_u = torch.zeros((num_nodes,), device=x.device)
            node_u.index_add_(0, src, diag.sum(dim=-1))
            Tt = 1.0 + F.softplus(self.a_u * node_u + self.b_u)
            logb = torch.log(b.clamp(min=EPS))
            probs_bp = torch.softmax(logb / Tt.view(-1, 1), dim=-1)
            extras["node_u"] = node_u
            extras["T"] = Tt

        if inf_cfg.use_cert:
            u_star = self._centered_log(m).detach()
            delta = self.estimate_delta_power(
                u_star=u_star,
                log_phi=log_phi.detach(),
                K=K.detach(),
                edge_index=edge_index,
                rev=rev,
                num_nodes=num_nodes,
                k=inf_cfg.cert_k,
                eps_fd=inf_cfg.cert_eps_fd,
                alpha_scale=inf_cfg.alpha_scale,
            )
            extras["delta"] = torch.tensor(delta, device=x.device)

        unary_probs = torch.softmax(log_phi, dim=-1)
        mix = torch.sigmoid(self.mix_logit)
        probs = (1.0 - mix) * probs_bp + mix * unary_probs
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=EPS)

        return probs, extras


def nll_with_label_smoothing(probs: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    logp = torch.log(probs.clamp(min=EPS))
    nll = -logp.gather(1, y.view(-1, 1)).mean()
    if eps <= 0:
        return nll
    smooth = -logp.mean(dim=1).mean()
    return (1.0 - eps) * nll + eps * smooth


@torch.no_grad()
def rw_norm(edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
    dst = edge_index[1]
    deg = torch.zeros((num_nodes,), device=device)
    deg.index_add_(0, dst, torch.ones((dst.numel(),), device=device))
    deg = deg.clamp(min=1.0)
    return 1.0 / deg[dst]


@torch.no_grad()
def ppr_propagate(edge_index: torch.Tensor, x0: torch.Tensor, num_nodes: int, steps: int, alpha: float) -> torch.Tensor:
    device = x0.device
    edge_index2, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    src, dst = edge_index2[0], edge_index2[1]
    norm = rw_norm(edge_index2, num_nodes, device).view(-1, 1)
    x = x0
    a = float(alpha)
    for _ in range(int(steps)):
        out = torch.zeros_like(x)
        out.index_add_(0, dst, x[src] * norm)
        x = (1.0 - a) * out + a * x0
    return x


@torch.no_grad()
def correct_and_smooth(
    probs: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    num_classes: int,
    correct_steps: int,
    correct_alpha: float,
    smooth_steps: int,
    smooth_alpha: float,
) -> torch.Tensor:
    device = probs.device
    Y = F.one_hot(y, num_classes=num_classes).float().to(device)

    E = torch.zeros_like(probs)
    E[train_mask] = Y[train_mask] - probs[train_mask]
    E = ppr_propagate(edge_index, E, num_nodes, correct_steps, correct_alpha)
    Z = probs + E
    Z = Z.clamp(min=0.0)
    Z = Z / Z.sum(dim=-1, keepdim=True).clamp(min=EPS)

    edge_index2, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    src, dst = edge_index2[0], edge_index2[1]
    norm = rw_norm(edge_index2, num_nodes, device).view(-1, 1)

    base = Z
    a = float(smooth_alpha)
    for _ in range(int(smooth_steps)):
        out = torch.zeros_like(Z)
        out.index_add_(0, dst, Z[src] * norm)
        Z = (1.0 - a) * out + a * base
        Z[train_mask] = Y[train_mask]

    Z = Z.clamp(min=0.0)
    Z = Z / Z.sum(dim=-1, keepdim=True).clamp(min=EPS)
    return Z


@torch.no_grad()
def tune_cs_params(
    probs: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    num_classes: int,
) -> Tuple[int, float, int, float]:
    correct_steps_list = [10, 20, 30, 50, 80]
    correct_alpha_list = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    smooth_steps_list = [10, 20, 30, 50, 80]
    smooth_alpha_list = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    best_cs = 50
    best_ca = 0.5
    best_ss = 50
    best_sa = 0.8
    best_acc = -1.0
    for cs in correct_steps_list:
        for ca in correct_alpha_list:
            for ss in smooth_steps_list:
                for sa in smooth_alpha_list:
                    out = correct_and_smooth(
                        probs,
                        y,
                        train_mask,
                        edge_index,
                        num_nodes,
                        num_classes,
                        cs,
                        ca,
                        ss,
                        sa,
                    )
                    acc = accuracy(out, y, val_mask)
                    if acc > best_acc + 1e-12:
                        best_acc = acc
                        best_cs, best_ca, best_ss, best_sa = cs, ca, ss, sa
    return best_cs, best_ca, best_ss, best_sa


@torch.no_grad()
def predict_probs(model: nn.Module, data, inf_cfg: InferenceConfig, mc_samples: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if mc_samples <= 1:
        model.eval()
        return model(data, inf_cfg)

    probs_sum = None
    extras_last: Dict[str, torch.Tensor] = {}
    model.train()
    for _ in range(mc_samples):
        p, ex = model(data, inf_cfg)
        probs_sum = p if probs_sum is None else probs_sum + p
        extras_last = ex
    probs = probs_sum / float(mc_samples)
    model.eval()
    return probs, extras_last


def train_one_run(args) -> None:
    device = torch.device(args.device)

    data, in_dim, num_classes = load_dataset(args.dataset, root=args.root)

    edge_index, _ = remove_self_loops(data.edge_index)
    edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)
    data.edge_index = edge_index
    data.rev_edge = compute_reverse_edge(data.edge_index, data.num_nodes)
    data = data.to(device)

    train_mask, val_mask, test_mask = get_split_masks(data, args.split)

    post = args.post
    if post == "auto":
        nm = args.dataset.lower()
        post = "cs" if nm in ["cora", "citeseer", "pubmed"] else "none"

    model = CertBP(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        edge_hidden_dim=args.edge_hidden_dim,
        dropout=args.dropout,
        w_max=args.w_max,
        cmp_margin=args.cmp_margin,
        encoder_type=args.encoder,
        gat_heads=args.gat_heads,
        alpha_max=args.alpha_max,
        mix_init=args.mix_init,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    inf_cfg_train = InferenceConfig(
        T=args.T,
        eta=args.eta,
        init_unary=not args.init_uniform,
        adaptive_outer=args.adaptive_outer,
        eta_min=args.eta_min,
        eta_max=args.eta_max,
        use_cert=args.use_cert,
        cert_k=args.cert_k,
        cert_eps_fd=args.cert_eps_fd,
        use_uncert=args.use_uncert_train,
        hutch_samples=args.hutch_samples,
        hutch_eps_fd=args.hutch_eps_fd,
        alpha_scale=1.0,
    )

    inf_cfg_eval = InferenceConfig(
        T=args.T_eval,
        eta=args.eta,
        init_unary=not args.init_uniform,
        adaptive_outer=args.adaptive_outer,
        eta_min=args.eta_min,
        eta_max=args.eta_max,
        use_cert=args.use_cert,
        cert_k=args.cert_k,
        cert_eps_fd=args.cert_eps_fd,
        use_uncert=args.use_uncert,
        hutch_samples=args.hutch_samples,
        hutch_eps_fd=args.hutch_eps_fd,
        alpha_scale=1.0,
    )

    best_val = -1.0
    best_state = None
    patience = 0

    for epoch in range(1, args.epochs + 1):
        inf_cfg_train.alpha_scale = 0.0 if epoch <= args.warmup else 1.0

        model.train()
        opt.zero_grad()

        probs, _ = model(data, inf_cfg_train)

        nll = nll_with_label_smoothing(probs[train_mask], data.y[train_mask], args.label_smoothing)

        if args.lambda_brier > 0:
            y_onehot = F.one_hot(data.y[train_mask], num_classes=num_classes).float()
            brier = ((probs[train_mask] - y_onehot) ** 2).sum(dim=-1).mean()
        else:
            brier = torch.zeros([], device=device)

        if args.lambda_cmp > 0:
            cmp_pen = model.compatibility_prior().mean()
        else:
            cmp_pen = torch.zeros([], device=device)

        if args.lambda_alpha > 0:
            alpha_pen = model._msg_alpha(inf_cfg_train.alpha_scale).pow(2)
        else:
            alpha_pen = torch.zeros([], device=device)

        loss = nll + args.lambda_brier * brier + args.lambda_cmp * cmp_pen + args.lambda_alpha * alpha_pen
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        probs_eval, _ = predict_probs(model, data, inf_cfg_eval, args.mc_eval)

        tr_acc = accuracy(probs_eval, data.y, train_mask)
        va_acc = accuracy(probs_eval, data.y, val_mask)
        te_acc = accuracy(probs_eval, data.y, test_mask)
        va_ece = ece_score(probs_eval, data.y, val_mask, n_bins=15)

        va_metric = va_acc
        te_metric = te_acc

        if post == "cs":
            probs_eval_cs = correct_and_smooth(
                probs_eval,
                data.y,
                train_mask,
                data.edge_index,
                data.num_nodes,
                num_classes,
                args.cs_correct_steps,
                args.cs_correct_alpha,
                args.cs_smooth_steps,
                args.cs_smooth_alpha,
            )
            va_metric = accuracy(probs_eval_cs, data.y, val_mask)
            te_metric = accuracy(probs_eval_cs, data.y, test_mask)

        improved = va_metric > best_val + 1e-12
        if improved:
            best_val = va_metric
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if epoch % args.log_every == 0 or epoch == 1:
            alpha_val = model._msg_alpha(1.0).detach().item()
            mix_val = torch.sigmoid(model.mix_logit).detach().item()
            print(
                f"[{args.dataset} split={args.split} seed={args.seed}] "
                f"ep={epoch:04d} loss={loss.item():.4f} nll={nll.item():.4f} "
                f"tr={tr_acc*100:.2f} va={va_acc*100:.2f} te={te_acc*100:.2f} vaECE={va_ece:.4f} "
                f"alpha={alpha_val:.3f} mix={mix_val:.3f} post={post} vaM={va_metric*100:.2f} teM={te_metric*100:.2f}"
            )

        if patience >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    probs_final, _ = predict_probs(model, data, inf_cfg_eval, args.mc_eval)
    probs_out = probs_final
    cs = args.cs_correct_steps
    ca = args.cs_correct_alpha
    ss = args.cs_smooth_steps
    sa = args.cs_smooth_alpha

    if post == "cs":
        cs, ca, ss, sa = tune_cs_params(
            probs_final,
            data.y,
            train_mask,
            val_mask,
            data.edge_index,
            data.num_nodes,
            num_classes,
        )
        probs_out = correct_and_smooth(
            probs_final,
            data.y,
            train_mask,
            data.edge_index,
            data.num_nodes,
            num_classes,
            cs,
            ca,
            ss,
            sa,
        )

    te_acc = accuracy(probs_out, data.y, test_mask)
    te_ece = ece_score(probs_out, data.y, test_mask, n_bins=15)
    te_nll = F.nll_loss(torch.log(probs_out[test_mask].clamp(min=EPS)), data.y[test_mask]).item()
    y_onehot = F.one_hot(data.y[test_mask], num_classes=num_classes).float()
    te_brier = ((probs_out[test_mask] - y_onehot) ** 2).sum(dim=-1).mean().item()

    print(
        f"\n[FINAL] {args.dataset} split={args.split} seed={args.seed} "
        f"TEST Acc={te_acc*100:.2f} ECE={te_ece:.4f} NLL={te_nll:.4f} Brier={te_brier:.4f} post={post} "
        f"cs_correct_steps={cs} cs_correct_alpha={ca} cs_smooth_steps={ss} cs_smooth_alpha={sa}"
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--root", type=str, default="./data")
    p.add_argument("--split", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--encoder", type=str, default="gat", choices=["mlp", "gcn", "gat"])
    p.add_argument("--gat_heads", type=int, default=8)

    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--edge_hidden_dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.6)
    p.add_argument("--w_max", type=float, default=0.8)
    p.add_argument("--cmp_margin", type=float, default=0.05)
    p.add_argument("--alpha_max", type=float, default=1.5)
    p.add_argument("--mix_init", type=float, default=0.995)

    p.add_argument("--T", type=int, default=10)
    p.add_argument("--T_eval", type=int, default=20)
    p.add_argument("--eta", type=float, default=0.2)
    p.add_argument("--init_uniform", action="store_true")

    p.add_argument("--adaptive_outer", type=int, default=1)
    p.add_argument("--eta_min", type=float, default=0.05)
    p.add_argument("--eta_max", type=float, default=0.8)

    p.add_argument("--use_cert", action="store_true")
    p.add_argument("--cert_k", type=int, default=10)
    p.add_argument("--cert_eps_fd", type=float, default=1e-3)

    p.add_argument("--use_uncert", action="store_true")
    p.add_argument("--use_uncert_train", action="store_true")
    p.add_argument("--hutch_samples", type=int, default=4)
    p.add_argument("--hutch_eps_fd", type=float, default=1e-3)

    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--lambda_brier", type=float, default=0.0)
    p.add_argument("--lambda_cmp", type=float, default=0.0)
    p.add_argument("--lambda_alpha", type=float, default=1e-3)

    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--mc_eval", type=int, default=1)

    p.add_argument("--post", type=str, default="auto", choices=["auto", "none", "cs"])
    p.add_argument("--cs_correct_steps", type=int, default=50)
    p.add_argument("--cs_correct_alpha", type=float, default=0.5)
    p.add_argument("--cs_smooth_steps", type=int, default=50)
    p.add_argument("--cs_smooth_alpha", type=float, default=0.8)

    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--patience", type=int, default=300)
    p.add_argument("--log_every", type=int, default=50)

    args = p.parse_args()

    set_seed(args.seed)
    train_one_run(args)


if __name__ == "__main__":
    main()
