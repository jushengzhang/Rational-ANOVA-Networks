import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# ==============================================================================
# ⚙️ Lightweight Configuration
# ==============================================================================
CONFIG = {
    'data_root': './data/denoising_dataset', 
    'patch_size': 4,               # 4x4 Patch -> 48 Input Dim -> 1128 Interaction Pairs
    'batch_size': 32,               
    'epochs': 100,                  
    'lr': 1e-3,                    
    'rational_hidden': 16,         # Hidden units for parameter compression
    'crop_size': 128,              # Training crop size (GPU Memory Safe)
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# ==============================================================================
# 1. Dataset Class (with Random Cropping)
# ==============================================================================
class DenoisingDataset(Dataset):
    def __init__(self, root_dir, crop_size=128, is_train=True):
        self.noisy_path = os.path.join(root_dir, 'real')
        self.gt_path = os.path.join(root_dir, 'mean')
        self.image_names = [f for f in os.listdir(self.noisy_path) 
                            if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        self.to_tensor = transforms.ToTensor()
        self.crop_size = crop_size
        self.is_train = is_train
        print(f"📁 [{'Train' if is_train else 'Test'}] Found {len(self.image_names)} images")

    def __len__(self): return len(self.image_names)

    def __getitem__(self, idx):
        try:
            name = self.image_names[idx]
            n_p = os.path.join(self.noisy_path, name)
            g_p = os.path.join(self.gt_path, name)
            
            # Load images
            noisy = cv2.cvtColor(cv2.imread(n_p), cv2.COLOR_BGR2RGB)
            gt = cv2.cvtColor(cv2.imread(g_p), cv2.COLOR_BGR2RGB)
            
            # Training: Random Crop | Testing: Center Crop
            h, w, _ = noisy.shape
            if h > self.crop_size and w > self.crop_size:
                if self.is_train:
                    top = np.random.randint(0, h - self.crop_size)
                    left = np.random.randint(0, w - self.crop_size)
                else:
                    top = (h - self.crop_size) // 2
                    left = (w - self.crop_size) // 2
                
                noisy = noisy[top:top+self.crop_size, left:left+self.crop_size]
                gt = gt[top:top+self.crop_size, left:left+self.crop_size]
            
            return self.to_tensor(noisy), self.to_tensor(gt)
            
        except Exception:
            return torch.zeros(3, self.crop_size, self.crop_size), torch.zeros(3, self.crop_size, self.crop_size)

# ==============================================================================
# 2. Rational Network Modules
# ==============================================================================
class RationalLayer1D_Dynamic(nn.Module):
    def __init__(self, in_features, out_features, degree_P=3, degree_Q=2, num_heads=1):
        super().__init__()
        self.degree_P, self.degree_Q = degree_P, degree_Q
        terms_P, terms_Q = degree_P + 1, degree_Q + 1
        self.p_coeffs = nn.Parameter(torch.randn(out_features, num_heads, in_features, terms_P) * 0.01)
        self.q_coeffs = nn.Parameter(torch.randn(out_features, num_heads, in_features, terms_Q) * 0.01)
        with torch.no_grad(): 
            self.q_coeffs[..., 0] = 1.0
            self.p_coeffs[..., 0] += 0.1

    def forward(self, x):
        powers_P = torch.arange(self.degree_P + 1, device=x.device).float()
        powers_Q = torch.arange(self.degree_Q + 1, device=x.device).float()
        basis_P = x.unsqueeze(1).unsqueeze(1).unsqueeze(-1).pow(powers_P)
        basis_Q = x.unsqueeze(1).unsqueeze(1).unsqueeze(-1).pow(powers_Q)
        P = (basis_P * self.p_coeffs.unsqueeze(0)).sum(dim=-1)
        Q = (basis_Q * self.q_coeffs.unsqueeze(0)).sum(dim=-1)
        return (P / (torch.abs(Q) + 1e-5)).sum(dim=2).sum(dim=-1)

class RationalLayer2D_Dynamic(nn.Module):
    def __init__(self, in_features, out_features, degree_P=3, degree_Q=2, num_heads=1):
        super().__init__()
        self.pairs = list(itertools.combinations(range(in_features), 2))
        self.num_pairs = len(self.pairs)
        self.num_heads = num_heads
        if self.num_pairs > 0:
            self.register_buffer('pow_P', self._generate_powers(degree_P))
            self.register_buffer('pow_Q', self._generate_powers(degree_Q))
            terms_P, terms_Q = len(self.pow_P), len(self.pow_Q)
            self.p_coeffs = nn.Parameter(torch.randn(out_features, num_heads, self.num_pairs, terms_P) * 0.01)
            self.q_coeffs = nn.Parameter(torch.randn(out_features, num_heads, self.num_pairs, terms_Q) * 0.01)
            with torch.no_grad(): self.q_coeffs[..., 0] = 1.0

    def _generate_powers(self, degree):
        powers = []
        for d in range(degree + 1):
            for i in range(d + 1):
                powers.append([i, d - i])
        return torch.tensor(powers, dtype=torch.float32)

    def forward(self, x):
        if self.num_pairs == 0: return 0.0
        idx_1 = [p[0] for p in self.pairs]; idx_2 = [p[1] for p in self.pairs]
        x1_exp = x[:, idx_1].unsqueeze(1).unsqueeze(1).unsqueeze(-1)
        x2_exp = x[:, idx_2].unsqueeze(1).unsqueeze(1).unsqueeze(-1)
        basis_P = (x1_exp ** self.pow_P[:, 0]) * (x2_exp ** self.pow_P[:, 1])
        basis_Q = (x1_exp ** self.pow_Q[:, 0]) * (x2_exp ** self.pow_Q[:, 1])
        P = (basis_P * self.p_coeffs.unsqueeze(0)).sum(dim=-1)
        Q = (basis_Q * self.q_coeffs.unsqueeze(0)).sum(dim=-1)
        return (P / (torch.abs(Q) + 1e-5)).sum(dim=2).sum(dim=2)

# --- Residual ANOVA Network with Local and Global Residuals ---
class ResidualRationalANOVA(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # Layer 1: Feature expansion/projection
        self.layer1_1d = RationalLayer1D_Dynamic(input_dim, hidden_dim, num_heads=2)
        self.layer1_2d = RationalLayer2D_Dynamic(input_dim, hidden_dim, num_heads=2)
        self.bn1 = nn.LayerNorm(hidden_dim)

        # Layer 2: Residual Block (hidden -> hidden)
        self.layer2_1d = RationalLayer1D_Dynamic(hidden_dim, hidden_dim, num_heads=2)
        self.layer2_2d = RationalLayer2D_Dynamic(hidden_dim, hidden_dim, num_heads=2) 
        self.bn2 = nn.LayerNorm(hidden_dim)

        # Layer 3: Output Projection
        self.layer3_1d = RationalLayer1D_Dynamic(hidden_dim, output_dim, num_heads=2)
        
    def forward(self, x):
        identity_global = x 

        # --- Layer 1 ---
        out1 = self.layer1_1d(x) + self.layer1_2d(x)
        h1 = self.bn1(out1)
        
        # --- Layer 2: Local Residual (h2 = f(h1) + h1) ---
        out2 = self.layer2_1d(h1) + self.layer2_2d(h1)
        h2 = self.bn2(out2 + h1) 
        
        # --- Layer 3: Predict Noise Residual ---
        noise_residual = self.layer3_1d(h2)
        
        # --- Global Residual: Final Image = Noisy Input + Learned Residual ---
        if identity_global.shape == noise_residual.shape:
            return identity_global + noise_residual
        return noise_residual

# ==============================================================================
# 3. Baseline MLP
# ==============================================================================
class StandardMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.BatchNorm1d(hidden_dim),       
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),       
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim) 
        )
    def forward(self, x): return self.net(x)

# ==============================================================================
# 4. Utilities
# ==============================================================================
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def solve_mlp_width(target_params, input_dim, output_dim):
    """Solves for hidden width to match parameter count of a target model."""
    a, b, c = 1, input_dim + output_dim + 6, output_dim - target_params
    delta = b**2 - 4*a*c
    if delta < 0: return 100 
    return int((-b + math.sqrt(delta)) / (2*a))

# ==============================================================================
# 5. Training Engine (with Chunking for Memory Efficiency)
# ==============================================================================
def train_engine(model, train_loader, test_loader, device, patch_size):
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.L1Loss()
    input_dim = 3 * patch_size * patch_size
    best_psnr = 0.0
    CHUNK_SIZE = 4096 # Process patches in small chunks to prevent VRAM overflow
    
    model.to(device)
    pbar = tqdm(range(CONFIG['epochs']), desc="Status")
    
    for epoch in pbar:
        model.train()
        epoch_loss = 0
        for noisy_img, gt_img in train_loader:
            noisy_img, gt_img = noisy_img.to(device), gt_img.to(device)
            
            # Extract patches using Unfold
            n_patches = torch.nn.functional.unfold(noisy_img, patch_size, stride=patch_size)
            g_patches = torch.nn.functional.unfold(gt_img, patch_size, stride=patch_size)
            inp = n_patches.transpose(1, 2).contiguous().view(-1, input_dim)
            tgt = g_patches.transpose(1, 2).contiguous().view(-1, input_dim)
            
            # Stochastic training
            perm = torch.randperm(inp.size(0))
            inp, tgt = inp[perm], tgt[perm]
            
            optimizer.zero_grad()
            batch_loss = 0
            
            # Chunking loop for backprop
            for c_in, c_gt in zip(torch.split(inp, CHUNK_SIZE), torch.split(tgt, CHUNK_SIZE)):
                pred = model(c_in)
                loss = criterion(pred, c_gt)
                loss.backward()
                batch_loss += loss.item() * len(c_in)
            
            optimizer.step()
            epoch_loss += batch_loss / len(inp)
            
        # Evaluation Phase
        model.eval()
        psnrs = []
        with torch.no_grad():
            for n_img, g_img in test_loader:
                n_img, g_img = n_img.to(device), g_img.to(device)
                b, c, h, w = n_img.shape
                
                patches = torch.nn.functional.unfold(n_img, patch_size, stride=patch_size)
                inp = patches.transpose(1, 2).contiguous().view(-1, input_dim)
                
                out_list = [model(c_in) for c_in in torch.split(inp, CHUNK_SIZE)]
                out = torch.cat(out_list, dim=0)
                
                out = out.view(b, -1, input_dim).transpose(1, 2)
                recon = torch.nn.functional.fold(out, (h, w), patch_size, stride=patch_size)
                recon = torch.clamp(recon, 0, 1)
                
                mse = torch.mean((recon - g_img) ** 2)
                if mse > 0:
                    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
                    psnrs.append(psnr.item())
        
        avg_psnr = np.mean(psnrs) if psnrs else 0
        if avg_psnr > best_psnr: best_psnr = avg_psnr
        pbar.set_postfix({'Loss': f"{epoch_loss/len(train_loader):.4f}", 'Best PSNR': f"{best_psnr:.2f}"})
        
    return best_psnr

# ==============================================================================
# Main Execution
# ==============================================================================
def main():
    print(f"📊 Dataset Path: {CONFIG['data_root']}")
    
    train_ds = DenoisingDataset(CONFIG['data_root'], CONFIG['crop_size'], is_train=True)
    test_ds = DenoisingDataset(CONFIG['data_root'], CONFIG['crop_size'], is_train=False)
    
    train_dl = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    patch_dim = 3 * CONFIG['patch_size'] ** 2
    print(f"🎯 Configuration: Patch Size={CONFIG['patch_size']} (Input Dim={patch_dim})")

    # Initialize Rational ANOVA Network
    net_r = ResidualRationalANOVA(patch_dim, CONFIG['rational_hidden'], patch_dim)
    p_r = count_params(net_r)
    print(f"\n[1] Rational Net | Hidden={CONFIG['rational_hidden']} | Params: {p_r:,}")

    # Initialize Parameter-matched MLP
    mlp_w = solve_mlp_width(p_r, patch_dim, patch_dim)
    net_m = StandardMLP(patch_dim, mlp_w, patch_dim)
    p_m = count_params(net_m)
    print(f"[2] MLP Baseline | Width={mlp_w} | Params: {p_m:,}")

    # Training Routine
    print(f"\n🚀 Training Rational Net...")
    psnr_r = train_engine(net_r, train_dl, test_dl, CONFIG['device'], CONFIG['patch_size'])
    
    print(f"\n🚀 Training MLP Baseline...")
    psnr_m = train_engine(net_m, train_dl, test_dl, CONFIG['device'], CONFIG['patch_size'])
    
    # Results Comparison
    print("\n" + "="*45)
    print(f"{'Model Variant':<20} | {'Param Count':<12} | {'PSNR (dB)':<10}")
    print("-" * 45)
    print(f"{'Rational ANOVA':<20} | {p_r:<12,} | {psnr_r:.2f} dB")
    print(f"{'Standard MLP':<20} | {p_m:<12,} | {psnr_m:.2f} dB")
    print("="*45)

if __name__ == '__main__':
    main()