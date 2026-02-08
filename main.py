import argparse
import copy
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
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, to_undirected

EPS = 1e-12


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


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


@torch.no_grad()
def nll_loss(probs: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    return F.nll_loss(torch.log(probs[mask].clamp(min=EPS)), y[mask]).item()


def get_split_masks(data, split_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    if train_mask.dim() == 1:
        return train_mask, val_mask, test_mask
    if split_idx < 0 or split_idx >= train_mask.size(1):
        raise ValueError(f"split_idx={split_idx} out of range for mask shape {train_mask.shape}")
    return train_mask[:, split_idx], val_mask[:, split_idx], test_mask[:, split_idx]


def dedup_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    src, dst = edge_index[0], edge_index[1]
    key = src * num_nodes + dst
    key_sorted, perm = torch.sort(key)
    src, dst = src[perm], dst[perm]
    mask = torch.ones_like(key_sorted, dtype=torch.bool)
    mask[1:] = key_sorted[1:] != key_sorted[:-1]
    return torch.stack([src[mask], dst[mask]], dim=0)


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


@torch.no_grad()
def dropedge_undirected(
    edge_index: torch.Tensor,
    rev_edge: torch.Tensor,
    drop_prob: float,
    num_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if drop_prob <= 0:
        return edge_index, rev_edge

    device = edge_index.device
    E = edge_index.size(1)
    idx = torch.arange(E, device=device)
    rev = rev_edge
    rep = torch.minimum(idx, rev)

    uniq_rep, inv = torch.unique(rep, sorted=False, return_inverse=True)

    keep_pair = (torch.rand(uniq_rep.size(0), device=device) > drop_prob)

    is_self = (edge_index[0] == edge_index[1])
    if is_self.any():
        self_pair_ids = torch.unique(inv[is_self])
        keep_pair[self_pair_ids] = True

    keep = keep_pair[inv]
    ei = edge_index[:, keep]

    rev2 = compute_reverse_edge(ei, num_nodes)
    return ei, rev2


def load_dataset(name: str, root: str = "./data", normalize_features: bool = True):
    name_raw = name
    name = name.lower()

    transform = NormalizeFeatures() if normalize_features else None
    is_webkb = name in ["texas", "wisconsin", "cornell"]

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

    if is_webkb:
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
        input_dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.dropout = dropout
        self.input_dropout = float(input_dropout)
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.gat_heads = gat_heads

        if encoder_type == "gcn":
            self.conv1 = GCNConv(in_dim, hidden_dim, cached=False, normalize=True)
            self.conv2 = GCNConv(hidden_dim, num_classes, cached=False, normalize=True)
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
        x = F.dropout(x, p=self.input_dropout, training=self.training)

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
    alpha_scale: float = 1.0


class CertBP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        edge_hidden_dim: int = 64,
        dropout: float = 0.6,
        input_dropout: float = 0.0,
        w_max: float = 0.8,
        cmp_margin: float = 0.05,
        encoder_type: str = "gat",
        gat_heads: int = 8,
        alpha_max: float = 1.5,
        mix_init: float = 0.6,
        msg_logit_init: float = -1.0,
        exp_clip: float = 20.0,
        feature_noise_std: float = 0.0,
        unary_temp: float = 1.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.w_max = w_max
        self.cmp_margin = cmp_margin
        self.alpha_max = alpha_max
        self.exp_clip = float(exp_clip)
        self.feature_noise_std = float(feature_noise_std)
        self.unary_temp = float(max(unary_temp, 1e-3))

        self.encoder = UnaryEncoder(
            in_dim, hidden_dim, num_classes, dropout,
            encoder_type=encoder_type, gat_heads=gat_heads, input_dropout=input_dropout
        )
        self.edge_mlp = MLP(in_dim=2 * hidden_dim + 2, hidden_dim=edge_hidden_dim, out_dim=1, dropout=dropout)

        self.R_raw = nn.Parameter(torch.empty(num_classes, num_classes))
        nn.init.xavier_uniform_(self.R_raw)

        self.R_scale_log = nn.Parameter(torch.tensor(0.0))
        self.msg_logit = nn.Parameter(torch.tensor(float(msg_logit_init)))

        mix_init = float(min(max(mix_init, 1e-4), 1.0 - 1e-4))
        self.mix_logit = nn.Parameter(torch.tensor(np.log(mix_init / (1.0 - mix_init)), dtype=torch.float))

    def _msg_alpha(self, alpha_scale: float = 1.0) -> torch.Tensor:
        return (self.alpha_max * torch.sigmoid(self.msg_logit)) * float(alpha_scale)

    def _sym_R(self) -> torch.Tensor:
        R = 0.5 * (self.R_raw + self.R_raw.t())
        scale = F.softplus(self.R_scale_log) + 1e-6
        return scale * torch.tanh(R)

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
        arg = w.view(-1, 1, 1) * R.view(1, R.size(0), R.size(1))
        arg = torch.clamp(arg, min=-self.exp_clip, max=self.exp_clip)
        K = torch.exp(arg)
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
            f = torch.nan_to_num(f, nan=1.0, posinf=1.0, neginf=1.0)
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
        f = torch.nan_to_num(f, nan=1.0, posinf=1.0, neginf=1.0)
        log_f = torch.log(f.clamp(min=EPS))
        log_f = log_f * edge_norm.view(-1, 1)

        sum_in = torch.zeros((num_nodes, C), device=device)
        sum_in.index_add_(0, dst, log_f)

        log_b = log_phi + alpha * sum_in
        return torch.softmax(log_b, dim=-1)

    def compatibility_prior(self) -> torch.Tensor:
        R = self._sym_R()
        diag = torch.diag(R).view(-1, 1)
        margin = self.cmp_margin
        M = R - diag + margin
        mask = 1.0 - torch.eye(R.size(0), device=R.device)
        return F.relu(M) * mask

    def forward(
        self,
        data,
        inf_cfg: InferenceConfig,
        edge_index: Optional[torch.Tensor] = None,
        rev: Optional[torch.Tensor] = None,
        x_override: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = data.x if x_override is None else x_override
        if self.training and self.feature_noise_std > 0:
            x = x + torch.randn_like(x) * self.feature_noise_std

        ei = data.edge_index if edge_index is None else edge_index
        r = data.rev_edge if rev is None else rev
        num_nodes = data.num_nodes

        h, logits = self.encoder(x, ei)
        log_phi = F.log_softmax(logits / self.unary_temp, dim=-1)

        struct = self._edge_struct_features(ei, num_nodes)
        w, K = self._build_pairwise_kernel(h, ei, struct, r)

        m = self._bp_one_run(
            log_phi=log_phi,
            K=K,
            edge_index=ei,
            rev=r,
            num_nodes=num_nodes,
            T=inf_cfg.T,
            eta=inf_cfg.eta,
            init_unary=inf_cfg.init_unary,
            alpha_scale=inf_cfg.alpha_scale,
        )

        b = self._compute_beliefs(log_phi, K, ei, num_nodes, m, alpha_scale=inf_cfg.alpha_scale)

        unary_probs = log_phi.exp()
        mix = torch.sigmoid(self.mix_logit)
        a = float(inf_cfg.alpha_scale)
        mix_eff = 1.0 - (1.0 - mix) * a
        probs = (1.0 - mix_eff) * b + mix_eff * unary_probs
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=EPS)

        extras: Dict[str, torch.Tensor] = {
            "beliefs": b,
            "messages": m,
            "alpha": self._msg_alpha(inf_cfg.alpha_scale).detach(),
            "mix": mix.detach(),
            "w": w if self.training else w.detach(),
        }
        return probs, extras


def nll_with_label_smoothing(probs: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    logp = torch.log(probs.clamp(min=EPS))
    nll = -logp.gather(1, y.view(-1, 1)).mean()
    if eps <= 0:
        return nll
    smooth = -logp.mean(dim=1).mean()
    return (1.0 - eps) * nll + eps * smooth


@torch.no_grad()
def predict_probs(
    model: nn.Module, data, inf_cfg: InferenceConfig, mc_samples: int,
    edge_index: Optional[torch.Tensor] = None,
    rev: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if mc_samples <= 1:
        model.eval()
        return model(data, inf_cfg, edge_index=edge_index, rev=rev)

    probs_sum = None
    extras_last: Dict[str, torch.Tensor] = {}
    model.train()
    for _ in range(mc_samples):
        p, ex = model(data, inf_cfg, edge_index=edge_index, rev=rev)
        probs_sum = p if probs_sum is None else probs_sum + p
        extras_last = ex
    probs = probs_sum / float(mc_samples)
    model.eval()
    return probs, extras_last


@torch.no_grad()
def apply_temperature_scaling(probs: torch.Tensor, T: float) -> torch.Tensor:
    T = float(max(T, 1e-3))
    logp = torch.log(probs.clamp(min=EPS))
    return torch.softmax(logp / T, dim=-1)


def fit_temperature(probs: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, max_iter: int = 50) -> float:
    device = probs.device
    logp = torch.log(probs[mask].clamp(min=EPS))
    yy = y[mask]

    logT = torch.zeros([], device=device, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.5, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        T = torch.exp(logT).clamp(min=1e-3, max=100.0)
        p = torch.softmax(logp / T, dim=-1)
        loss = F.nll_loss(torch.log(p.clamp(min=EPS)), yy)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(logT).detach().item())


def train_one_run(args) -> None:
    device = torch.device(args.device)
    set_seed(args.seed, deterministic=args.deterministic)

    data, in_dim, num_classes = load_dataset(args.dataset, root=args.root, normalize_features=args.normalize_features)

    edge_index, _ = remove_self_loops(data.edge_index)
    edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)
    edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=data.num_nodes)
    edge_index = dedup_edge_index(edge_index, data.num_nodes)

    data.edge_index = edge_index
    data.rev_edge = compute_reverse_edge(data.edge_index, data.num_nodes)
    data = data.to(device)

    train_mask, val_mask, test_mask = get_split_masks(data, args.split)

    nm = args.dataset.lower()
    is_webkb = nm in ["texas", "wisconsin", "cornell"]
    is_planetoid = nm in ["cora", "citeseer", "pubmed"]

    if args.auto_reg and is_webkb:
        if args.encoder == "auto":
            args.encoder = "mlp"
        args.label_smoothing = max(args.label_smoothing, 0.1)
        args.dropedge = max(args.dropedge, 0.2)
        args.weight_decay = max(args.weight_decay, 5e-3)
        args.lambda_alpha = max(args.lambda_alpha, 5e-3)
        args.lambda_mix = max(args.lambda_mix, 0.1)
        args.mix_target = 0.85
        args.w_max = min(args.w_max, 0.4)
        args.early_metric = "nll"
        if args.post == "auto":
            args.post = "ts"
    else:
        if args.encoder == "auto":
            args.encoder = "gat"
        if args.auto_reg and is_planetoid:
            args.label_smoothing = max(args.label_smoothing, 0.05)
            args.dropedge = max(args.dropedge, 0.2)
            args.input_dropout = max(args.input_dropout, 0.1)
            args.feature_noise = max(args.feature_noise, 0.01)
            args.lambda_mix = min(args.lambda_mix, 1e-3)
            args.mix_target = 0.8
            args.lambda_w = max(args.lambda_w, 1e-3)
            args.lambda_R = max(args.lambda_R, 1e-4)
            args.w_max = min(args.w_max, 0.6)
            args.exp_clip = min(args.exp_clip, 10.0)

    model = CertBP(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        edge_hidden_dim=args.edge_hidden_dim,
        dropout=args.dropout,
        input_dropout=args.input_dropout,
        w_max=args.w_max,
        cmp_margin=args.cmp_margin,
        encoder_type=args.encoder,
        gat_heads=args.gat_heads,
        alpha_max=args.alpha_max,
        mix_init=args.mix_init,
        msg_logit_init=args.msg_logit_init,
        exp_clip=args.exp_clip,
        feature_noise_std=args.feature_noise,
        unary_temp=args.unary_temp,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.use_cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs, eta_min=args.lr * args.cosine_min_lr_scale
        )

    inf_cfg_train = InferenceConfig(
        T=args.T,
        eta=args.eta,
        init_unary=not args.init_uniform,
        alpha_scale=1.0,
    )
    inf_cfg_eval = InferenceConfig(
        T=args.T_eval,
        eta=args.eta,
        init_unary=not args.init_uniform,
        alpha_scale=1.0,
    )

    best_score = None
    best_state = None
    patience = 0

    for epoch in range(1, args.epochs + 1):
        if args.warmup <= 0:
            inf_cfg_train.alpha_scale = 1.0
        else:
            inf_cfg_train.alpha_scale = min(1.0, float(epoch) / float(args.warmup))

        ei_train, rev_train = data.edge_index, data.rev_edge
        if args.dropedge > 0:
            ei_train, rev_train = dropedge_undirected(data.edge_index, data.rev_edge, args.dropedge, data.num_nodes)

        model.train()
        opt.zero_grad(set_to_none=True)

        probs, ex = model(data, inf_cfg_train, edge_index=ei_train, rev=rev_train)

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

        if args.lambda_mix > 0:
            mix = torch.sigmoid(model.mix_logit)
            mix_pen = (mix - args.mix_target).pow(2)
        else:
            mix_pen = torch.zeros([], device=device)

        if args.lambda_w > 0:
            w_pen = ex["w"].mean()
        else:
            w_pen = torch.zeros([], device=device)

        if args.lambda_R > 0:
            R_pen = model._sym_R().pow(2).mean()
        else:
            R_pen = torch.zeros([], device=device)

        loss = (
            nll
            + args.lambda_brier * brier
            + args.lambda_cmp * cmp_pen
            + args.lambda_alpha * alpha_pen
            + args.lambda_mix * mix_pen
            + args.lambda_w * w_pen
            + args.lambda_R * R_pen
        )

        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if scheduler is not None:
            scheduler.step()

        probs_eval, _ = predict_probs(model, data, inf_cfg_eval, args.mc_eval, edge_index=data.edge_index, rev=data.rev_edge)

        tr_acc = accuracy(probs_eval, data.y, train_mask)
        va_acc = accuracy(probs_eval, data.y, val_mask)
        te_acc = accuracy(probs_eval, data.y, test_mask)
        va_ece = ece_score(probs_eval, data.y, val_mask, n_bins=15)
        va_nll = nll_loss(probs_eval, data.y, val_mask)

        if args.early_metric == "acc":
            score = va_acc
            improved = (best_score is None) or (score > best_score + 1e-12)
        else:
            score = -va_nll
            improved = (best_score is None) or (score > best_score + 1e-12)

        if improved:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1

        if epoch % args.log_every == 0 or epoch == 1:
            alpha_val = model._msg_alpha(1.0).detach().item()
            mix_val = torch.sigmoid(model.mix_logit).detach().item()
            print(
                f"[{args.dataset} split={args.split} seed={args.seed}] "
                f"ep={epoch:04d} loss={loss.item():.4f} nll={nll.item():.4f} "
                f"tr={tr_acc*100:.2f} va={va_acc*100:.2f} te={te_acc*100:.2f} "
                f"vaNLL={va_nll:.4f} vaECE={va_ece:.4f} "
                f"alpha={alpha_val:.3f} mix={mix_val:.3f} post={args.post}"
            )

        if patience >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    probs_final, _ = predict_probs(model, data, inf_cfg_eval, args.mc_eval, edge_index=data.edge_index, rev=data.rev_edge)

    post = args.post
    T = 1.0
    if post == "ts":
        T = fit_temperature(probs_final, data.y, val_mask)
        probs_final = apply_temperature_scaling(probs_final, T)

    te_acc = accuracy(probs_final, data.y, test_mask)
    te_ece = ece_score(probs_final, data.y, test_mask, n_bins=15)
    te_nll = F.nll_loss(torch.log(probs_final[test_mask].clamp(min=EPS)), data.y[test_mask]).item()
    y_onehot = F.one_hot(data.y[test_mask], num_classes=num_classes).float()
    te_brier = ((probs_final[test_mask] - y_onehot) ** 2).sum(dim=-1).mean().item()

    extra_post = f"ts(T={T:.3f})" if post == "ts" else "none"
    print(
        f"\n[FINAL] {args.dataset} split={args.split} seed={args.seed} "
        f"TEST Acc={te_acc*100:.2f} ECE={te_ece:.4f} NLL={te_nll:.4f} Brier={te_brier:.4f} post={extra_post}"
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--root", type=str, default="./data")
    p.add_argument("--split", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--deterministic", action="store_true")

    p.add_argument("--auto_reg", dest="auto_reg", action="store_true", default=True)
    p.add_argument("--no_auto_reg", dest="auto_reg", action="store_false")

    p.add_argument("--normalize_features", dest="normalize_features", action="store_true", default=True)
    p.add_argument("--no_normalize_features", dest="normalize_features", action="store_false")

    p.add_argument("--use_cosine", dest="use_cosine", action="store_true", default=True)
    p.add_argument("--no_use_cosine", dest="use_cosine", action="store_false")

    p.add_argument("--encoder", type=str, default="auto", choices=["auto", "mlp", "gcn", "gat"])
    p.add_argument("--gat_heads", type=int, default=8)

    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--edge_hidden_dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.6)
    p.add_argument("--input_dropout", type=float, default=0.0)
    p.add_argument("--feature_noise", type=float, default=0.0)

    p.add_argument("--w_max", type=float, default=0.8)
    p.add_argument("--cmp_margin", type=float, default=0.05)
    p.add_argument("--alpha_max", type=float, default=1.5)
    p.add_argument("--mix_init", type=float, default=0.6)
    p.add_argument("--msg_logit_init", type=float, default=-1.0)
    p.add_argument("--exp_clip", type=float, default=20.0)
    p.add_argument("--unary_temp", type=float, default=1.5)

    p.add_argument("--T", type=int, default=10)
    p.add_argument("--T_eval", type=int, default=20)
    p.add_argument("--eta", type=float, default=0.2)
    p.add_argument("--init_uniform", action="store_true")

    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--lambda_brier", type=float, default=0.0)
    p.add_argument("--lambda_cmp", type=float, default=0.0)
    p.add_argument("--lambda_alpha", type=float, default=1e-3)
    p.add_argument("--lambda_mix", type=float, default=1e-2)
    p.add_argument("--mix_target", type=float, default=0.5)
    p.add_argument("--lambda_w", type=float, default=0.0)
    p.add_argument("--lambda_R", type=float, default=0.0)

    p.add_argument("--dropedge", type=float, default=0.0)

    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--mc_eval", type=int, default=1)

    p.add_argument("--post", type=str, default="auto", choices=["auto", "none", "ts"])
    p.add_argument("--early_metric", type=str, default="acc", choices=["acc", "nll"])

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--patience", type=int, default=300)
    p.add_argument("--log_every", type=int, default=50)

    p.add_argument("--cosine_min_lr_scale", type=float, default=0.1)

    args = p.parse_args()
    if args.post == "auto":
        args.post = "none"

    train_one_run(args)


if __name__ == "__main__":
    main()
