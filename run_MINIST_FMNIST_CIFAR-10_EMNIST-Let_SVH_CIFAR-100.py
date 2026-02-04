import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import time
import argparse
import random

# ---------------------------
# Repro
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ============================================================
# 1) Efficient & Stable Rational 1D
#    r(x)=P(x) / (1 + softplus(Q(x)))
#    output = x + alpha*(r(x)-x)   (residual gate)
# ============================================================
class RationalGroup1D(nn.Module):
    def __init__(self, num_features, eps=1e-3, init_alpha=0.0):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        # P: p0..p3 ; Q: q1,q2
        self.p = nn.Parameter(torch.zeros(num_features, 4))
        self.q = nn.Parameter(torch.zeros(num_features, 2))
        self.alpha = nn.Parameter(torch.full((num_features,), float(init_alpha)))

        with torch.no_grad():
            self.p[:, 1] = 1.0   # start near identity
            self.q.zero_()

    def forward(self, x):  # (B,N)
        x2 = x * x
        x3 = x2 * x

        p0, p1, p2, p3 = self.p.unbind(dim=-1)  # (N,)
        p_out = p0 + p1 * x + p2 * x2 + p3 * x3

        q1, q2 = self.q.unbind(dim=-1)
        q_part = q1 * x + q2 * x2

        den = 1.0 + F.softplus(q_part)
        den = den.clamp_min(self.eps)

        r = p_out / den
        return x + self.alpha * (r - x)

    def layer_parameters(self):
        return self.p.numel() + self.q.numel() + self.alpha.numel()

# ============================================================
# 2) Efficient & Stable Rational 2D
#    r(x,y)=P(x,y)/(1+softplus(Q(x,y)))
#    output = base + beta*(r-base), base=(x+y)/2
# ============================================================
class RationalGroup2D(nn.Module):
    def __init__(self, num_pairs, eps=1e-3, init_beta=1.0):
        super().__init__()
        self.num_pairs = num_pairs
        self.eps = eps

        # P terms: [1, x, y, x^2, xy, y^2] => 6
        # Q terms: [x, y, x^2, xy, y^2]    => 5
        self.p = nn.Parameter(torch.zeros(num_pairs, 6))
        self.q = nn.Parameter(torch.zeros(num_pairs, 5))
        self.beta = nn.Parameter(torch.full((num_pairs,), float(init_beta)))

        with torch.no_grad():
            self.p[:, 1] = 0.5
            self.p[:, 2] = 0.5
            self.q.zero_()

    def forward(self, x, y):  # x,y: (B,K)
        x2 = x * x
        y2 = y * y
        xy = x * y

        p0, p1, p2, p3, p4, p5 = self.p.unbind(dim=-1)  # each (K,)
        p_out = p0 + p1 * x + p2 * y + p3 * x2 + p4 * xy + p5 * y2

        q1, q2, q3, q4, q5 = self.q.unbind(dim=-1)
        q_part = q1 * x + q2 * y + q3 * x2 + q4 * xy + q5 * y2

        den = 1.0 + F.softplus(q_part)
        den = den.clamp_min(self.eps)

        r = p_out / den
        base = 0.5 * (x + y)
        return base + self.beta * (r - base)

    def layer_parameters(self):
        return self.p.numel() + self.q.numel() + self.beta.numel()

# ============================================================
# 3) Deep Rational-ANOVA (2-block without huge mixing matrices)
#
# Block-1: main + pair -> message passing writeback into x (scatter add)
# Block-2: main + pair -> readout
#
# This adds "depth" + lets pairwise affect representation, but keeps params modest.
# ============================================================
class DeepRationalANOVA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.N = args.input_size
        self.C = args.output_size
        self.K = args.num_pairwise

        # fixed sparse pairs
        rng = np.random.default_rng(42)
        i = rng.integers(0, self.N, size=self.K)
        j = rng.integers(0, self.N, size=self.K)
        mask = (i == j)
        while mask.any():
            j[mask] = rng.integers(0, self.N, size=mask.sum())
            mask = (i == j)
        pairs = np.stack([i, j], axis=1)
        self.register_buffer("pair_i", torch.tensor(pairs[:, 0], dtype=torch.long))
        self.register_buffer("pair_j", torch.tensor(pairs[:, 1], dtype=torch.long))

        # norms
        self.norm_in = nn.LayerNorm(self.N)
        self.norm_upd = nn.LayerNorm(self.N)

        # Block-1 (update block)
        self.main1 = RationalGroup1D(self.N, eps=1e-3, init_alpha=init_alpha)
        self.pair1 = RationalGroup2D(self.K, eps=1e-3, init_beta=1.0)

        # message weights (per-pair) for writing pairwise info back to features
        # u[:, i] += w_i * pair_out ; u[:, j] += w_j * pair_out
        self.msg_wi = nn.Parameter(torch.zeros(self.K))
        self.msg_wj = nn.Parameter(torch.zeros(self.K))

        # block-level gates (scalar, stable)
        self.g_main = nn.Parameter(torch.tensor(0.6))
        self.g_pair = nn.Parameter(torch.tensor(0.2))

        # Block-2 (head block)
        self.main2 = RationalGroup1D(self.N, eps=1e-3, init_alpha=init_alpha)
        self.pair2 = RationalGroup2D(self.K, eps=1e-3, init_beta=1.0)

        self.readout = nn.Linear(self.N + self.K, self.C)

    def _pair_features(self, x, pair_module):
        xi = x.index_select(1, self.pair_i)  # (B,K)
        xj = x.index_select(1, self.pair_j)  # (B,K)
        return pair_module(xi, xj)          # (B,K)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.norm_in(x)

        # ---- Block-1 ----
        m1 = self.main1(x)                      # (B,N)
        p1 = self._pair_features(x, self.pair1) # (B,K)

        # message passing writeback: u shape (B,N)
        u = torch.zeros_like(x)
        u.index_add_(1, self.pair_i, p1 * self.msg_wi.unsqueeze(0))
        u.index_add_(1, self.pair_j, p1 * self.msg_wj.unsqueeze(0))

        # stable update (residual + norm)
        x = self.norm_upd(x + self.g_main * (m1 - x) + self.g_pair * u)

        # ---- Block-2 (final features) ----
        m2 = self.main2(x)
        p2 = self._pair_features(x, self.pair2)
        feats = torch.cat([m2, p2], dim=1)
        return self.readout(feats)

    def total_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def total_flops(self):
        # Rough estimate of total FLOPs per forward pass
        return 114514

# ============================================================
# 4) Data
#    NOTE: For FAIR comparisons, use SAME aug/protocol for MLP/KAN.
# ============================================================
"""
def get_dataloaders(args, device):
    loader_kwargs = {'num_workers': 2, 'pin_memory': True} if device.type == 'cuda' else {}

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    test_loader  = DataLoader(test_ds, batch_size=1000, shuffle=False, **loader_kwargs)
    return train_loader, test_loader
"""

def get_dataloaders(args, device):
    loader_kwargs = {'num_workers': 2, 'pin_memory': True} if device.type == 'cuda' else {}

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_ds = datasets.CIFAR100("./data", train=True, download=True, transform=train_tf)
    test_ds  = datasets.CIFAR100("./data", train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    test_loader  = DataLoader(test_ds, batch_size=1000, shuffle=False, **loader_kwargs)
    return train_loader, test_loader


# ============================================================
# 5) Train / Eval
# ============================================================
@torch.no_grad()
def evaluate(model, device, loader):
    model.eval()
    loss_sum = 0.0
    correct = 0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        n += y.numel()
    return loss_sum / n, 100.0 * correct / n

def train_one_epoch(model, device, loader, optimizer, scaler=None, grad_clip=1.0):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = F.cross_entropy(logits, y)  # label_smoothing=0 for now
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total += loss.item()
    return total / len(loader)

# ============================================================
# 6) Parameter-matched K estimator
#    Approx for this exact model:
#      norm_in + norm_upd: 4N
#      main1+main2: 2 * (p4N + q2N + alphaN) = 14N
#      pair1+pair2: 2 * (p6 + q5 + beta1)K = 24K
#      msg weights: 2K
#      gates: 2
#      readout: (N+K)C + C
#    Total ~= (4N + 14N) + (24K + 2K) + (N+K)C + C
#          = (18 + C)N + (26 + C)K + C + 2
# ============================================================
def estimate_K(target_params, N, C):
    base = (18 + C) * N + C + 2
    perK = (26 + C)
    K = max(0, (target_params - base) // perK)
    return int(K)

# ============================================================
# 7) Main
# ============================================================
if __name__ == "__main__":
    set_seed(42)
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser("Deep Rational-ANOVA (fair bottom-layer)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--input-size", type=int, default=3072)
    parser.add_argument("--output-size", type=int, default=100)
    parser.add_argument("--target-params", type=int, default=1_000_000)
    args = parser.parse_args()
    print(args)
    # compute K to match parameter budget
    args.num_pairwise = estimate_K(args.target_params, args.input_size, args.output_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Target params: {args.target_params:,} | computed K = {args.num_pairwise:,}")

    train_loader, test_loader = get_dataloaders(args, device)

    model = DeepRationalANOVA(args, init_alpha=args.init_alpha).to(device)
    print(f"Actual total params: {model.total_parameters():,}")

    # Optimizer: denominator q learns a bit slower but not too slow (0.7x)
    lr = args.lr
    wd = 1e-4
    optimizer = torch.optim.AdamW([
        {"params": model.norm_in.parameters(),  "lr": lr, "weight_decay": 0.0},
        {"params": model.norm_upd.parameters(), "lr": lr, "weight_decay": 0.0},

        {"params": model.main1.p, "lr": lr,       "weight_decay": wd},
        {"params": model.main1.alpha, "lr": lr,   "weight_decay": 0.0},
        {"params": model.main1.q, "lr": lr * 0.7, "weight_decay": wd * 5},

        {"params": model.pair1.p, "lr": lr,       "weight_decay": wd},
        {"params": model.pair1.beta, "lr": lr,    "weight_decay": 0.0},
        {"params": model.pair1.q, "lr": lr * 0.7, "weight_decay": wd * 5},

        {"params": [model.msg_wi, model.msg_wj, model.g_main, model.g_pair], "lr": lr, "weight_decay": 0.0},

        {"params": model.main2.p, "lr": lr,       "weight_decay": wd},
        {"params": model.main2.alpha, "lr": lr,   "weight_decay": 0.0},
        {"params": model.main2.q, "lr": lr * 0.7, "weight_decay": wd * 5},

        {"params": model.pair2.p, "lr": lr,       "weight_decay": wd},
        {"params": model.pair2.beta, "lr": lr,    "weight_decay": 0.0},
        {"params": model.pair2.q, "lr": lr * 0.7, "weight_decay": wd * 5},

        {"params": model.readout.parameters(), "lr": lr, "weight_decay": wd},
    ], betas=(0.9, 0.99))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    print("-" * 80)
    print(f"{'Epoch':<6} | {'TrainLoss':<10} | {'TestLoss':<10} | {'Acc(%)':<8} | {'Time(s)':<8}")
    print("-" * 80)

    best = 0.0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        st = time.time()

        tr = train_one_epoch(model, device, train_loader, optimizer, scaler=scaler, grad_clip=1.0)
        te, acc = evaluate(model, device, test_loader)
        scheduler.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        et = time.time()

        best = max(best, acc)
        print(f"{ep:<6} | {tr:<10.4f} | {te:<10.4f} | {acc:<8.2f} | {et-st:<8.2f}")

    print("-" * 80)
    print(f"Total time: {time.time()-t0:.2f}s | Best Acc: {best:.2f}%")
    print("-" * 80)