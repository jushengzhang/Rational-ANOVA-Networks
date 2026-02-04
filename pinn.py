import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import copy

# ==========================================
# Configuration and Initialization
# ==========================================
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 1. Physics Environment
# ==========================================
def get_ground_truth(x, y):
    """Returns the analytical solution for validation."""
    return 1.0 / (1.0 + x**2 + y**2)

def get_forcing_function(x, y):
    """Returns the source term f for the Poisson equation."""
    denom = (1.0 + x**2 + y**2)
    f = 4.0 * (1.0 - x**2 - y**2) / (denom ** 3)
    return f

# ==========================================
# 2. Model Definitions
# ==========================================

# --- A. MLP (~5300 params) ---
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 1)
        )
    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))

# --- B. KAN (~5200 params) ---
class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        h = (1 + 1) / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * h - 1).float()
        self.register_buffer("grid", grid)
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features * (grid_size + spline_order)))
        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5) * scale_base)
        nn.init.uniform_(self.spline_weight, -scale_noise, scale_noise)
        self.scale_spline = scale_spline

    def b_splines(self, x):
        x = x.unsqueeze(-1)
        grid = self.grid
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()
        for k in range(1, self.spline_order + 1):
            bases = (x - grid[:-(k + 1)]) / (grid[k:-1] - grid[:-(k + 1)]) * bases[:, :, :-1] + \
                    (grid[k + 1:] - x) / (grid[k + 1:] - grid[1:-k]) * bases[:, :, 1:]
        return bases

    def forward(self, x):
        base_output = F.linear(F.silu(x), self.base_weight)
        batch_size = x.shape[0]
        spline_basis = self.b_splines(x).view(batch_size, -1)
        spline_output = F.linear(spline_basis, self.spline_weight)
        return base_output + self.scale_spline * spline_output

    def regularization_loss(self):
        return self.spline_weight.abs().mean()

class KAN(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = 24
        self.layer1 = KANLayer(2, hidden_dim)
        self.layer2 = KANLayer(hidden_dim, hidden_dim)
        self.layer3 = KANLayer(hidden_dim, 1)
    def forward(self, x, y):
        input_tensor = torch.cat([x, y], dim=1)
        x = self.layer1(input_tensor)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
    def get_reg_loss(self):
        return self.layer1.regularization_loss() + self.layer2.regularization_loss() + self.layer3.regularization_loss()

# --- C. Hybrid-Rational-ANOVA (72 Params) ---
class RationalLayer1D_Cubic(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 2
        self.num_terms = 4 
        self.p_coeffs = nn.Parameter(torch.zeros(1, self.num_heads, self.num_terms))
        self.q_coeffs = nn.Parameter(torch.zeros(1, self.num_heads, self.num_terms))
        with torch.no_grad():
            self.p_coeffs.data.normal_(0, 0.01)
            self.q_coeffs.data.fill_(0.0)
            self.q_coeffs[:, :, 0] = 1.0 

    def forward(self, z):
        z = z.unsqueeze(1)
        basis = torch.cat([torch.ones_like(z), z, z**2, z**3], dim=2)
        P = (basis * self.p_coeffs).sum(dim=-1)
        Q = (basis * self.q_coeffs).sum(dim=-1)
        out = P / (torch.abs(Q) + 1e-6)
        return out.sum(dim=1, keepdim=True) 

class RationalLayer2D_Cubic(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 2
        self.num_terms = 10
        self.p_coeffs = nn.Parameter(torch.zeros(1, self.num_heads, self.num_terms))
        self.q_coeffs = nn.Parameter(torch.zeros(1, self.num_heads, self.num_terms))
        with torch.no_grad():
            self.p_coeffs.data.normal_(0, 0.01)
            self.q_coeffs.data.fill_(0.0)
            self.q_coeffs[:, :, 0] = 1.0

    def forward(self, x, y):
        x = x.unsqueeze(1); y = y.unsqueeze(1)
        basis = torch.cat([
            torch.ones_like(x), x, y, x**2, x*y, y**2,
            x**3, (x**2)*y, x*(y**2), y**3
        ], dim=2)
        P = (basis * self.p_coeffs).sum(dim=-1)
        Q = (basis * self.q_coeffs).sum(dim=-1)
        out = P / (torch.abs(Q) + 1e-6)
        return out.sum(dim=1, keepdim=True)

class RationalANOVA(nn.Module):
    def __init__(self):
        super().__init__()
        self.main_x = RationalLayer1D_Cubic()
        self.main_y = RationalLayer1D_Cubic()
        self.interaction = RationalLayer2D_Cubic()
    def forward(self, x, y):
        return self.main_x(x) + self.main_y(y) + self.interaction(x, y)
    def get_reg_loss(self):
        l1 = torch.sum(torch.abs(self.main_x.p_coeffs)) + torch.sum(torch.abs(self.main_x.q_coeffs))
        l2 = torch.sum(torch.abs(self.main_y.p_coeffs)) + torch.sum(torch.abs(self.main_y.q_coeffs))
        l3 = torch.sum(torch.abs(self.interaction.p_coeffs)) + torch.sum(torch.abs(self.interaction.q_coeffs))
        return l1 + l2 + l3

# ==========================================
# 3. Training Script (NO REGULARIZATION)
# ==========================================
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_experiment_hybrid(model_name, total_epochs=2000, switch_epoch=500):
    # Model Initialization
    if model_name == "MLP":
        model = MLP().to(device)
    elif model_name == "KAN":
        model = KAN().to(device)
    else:
        model = RationalANOVA().to(device)
    
    # Regularization weight set to 0 to test pure expressivity
    lambda_reg = 0.0
        
    print(f"\n🚀 Model: {model_name} | Parameters: {count_params(model)} | Reg Weight: {lambda_reg}")
    
    # Stage 1: Adam Optimizer
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Stage 2: LBFGS Optimizer
    optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, 
                                        history_size=20, max_iter=20, 
                                        line_search_fn="strong_wolfe")
    
    loss_history = []
    best_loss = float('inf')
    best_model_weights = None
    
    start_time = time.time()
    
    for epoch in range(total_epochs):
        
        def closure():
            if epoch < switch_epoch:
                optimizer_adam.zero_grad()
            else:
                optimizer_lbfgs.zero_grad()
            
            # Resampling collocation points
            x_in = (torch.rand(2000, 1, device=device) * 2 - 1).requires_grad_(True)
            y_in = (torch.rand(2000, 1, device=device) * 2 - 1).requires_grad_(True)
            
            # Sampling boundary points
            x_bc = torch.rand(400, 1, device=device) * 2 - 1
            y_bc = torch.rand(400, 1, device=device) * 2 - 1
            mask = torch.rand(400, 1, device=device) > 0.5
            x_bc[mask] = torch.sign(x_bc[mask]); y_bc[~mask] = torch.sign(y_bc[~mask])
            
            # PDE Residual Loss
            u_pred = model(x_in, y_in)
            grads = torch.autograd.grad(u_pred, [x_in, y_in], torch.ones_like(u_pred), create_graph=True)
            u_xx = torch.autograd.grad(grads[0], x_in, torch.ones_like(grads[0]), create_graph=True)[0]
            u_yy = torch.autograd.grad(grads[1], y_in, torch.ones_like(grads[1]), create_graph=True)[0]
            
            loss_pde = torch.mean((- (u_xx + u_yy) - get_forcing_function(x_in, y_in)) ** 2)
            
            # Boundary Condition Loss
            loss_bc = torch.mean((model(x_bc, y_bc) - get_ground_truth(x_bc, y_bc))**2)
            
            # Regularization Loss
            reg = 0.0
            if model_name != "MLP":
                reg = model.get_reg_loss()
                
            loss = loss_pde + loss_bc + lambda_reg * reg
            loss.backward()
            return loss

        # Switching Logic
        if epoch < switch_epoch:
            loss = closure()
            optimizer_adam.step()
        else:
            loss = optimizer_lbfgs.step(closure)

        current_loss = loss.item()
        loss_history.append(current_loss)
        
        if current_loss < best_loss:
            best_loss = current_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: {current_loss:.6f} (Best: {best_loss:.6f})")

    # Restore Best Weights
    print(f"Restoring best model with loss: {best_loss:.8f}")
    model.load_state_dict(best_model_weights)

    duration = time.time() - start_time
    print(f"✅ {model_name} Training Completed in {duration:.2f}s")
    
    # Inspect Learned Coefficients
    if model_name == "Rational":
        print("\n[Hybrid-Rational 72 Params] Interaction Coefficients (Head 0, Normalized):")
        m = model.interaction
        p = m.p_coeffs[0, 0, :].detach().cpu().numpy()
        q = m.q_coeffs[0, 0, :].detach().cpu().numpy()
        norm = q[0] 
        p /= norm; q /= norm
        terms = ["1", "x", "y", "x^2", "xy", "y^2", "x^3", "x^2y", "xy^2", "y^3"]
        print(f"{'Term':<6} | {'P (Num)':<10} | {'Q (Den)':<10}")
        print("-" * 35)
        for i, t in enumerate(terms):
            print(f"{t:<6} | {p[i]:.4f}      | {q[i]:.4f}")

    return loss_history

# ==========================================
# 4. Execution
# ==========================================
# 2000 total epochs, switch to LBFGS at epoch 500
loss_mlp = train_experiment_hybrid("MLP", total_epochs=2000, switch_epoch=500)
loss_kan = train_experiment_hybrid("KAN", total_epochs=2000, switch_epoch=500)
loss_rat = train_experiment_hybrid("Rational", total_epochs=2000, switch_epoch=500)

plt.figure(figsize=(10, 6))
plt.semilogy(loss_mlp, label='MLP (~5300 params)', color='gray', linestyle='--')
plt.semilogy(loss_kan, label='KAN (~5200 params)', color='blue', linestyle='-.')
plt.semilogy(loss_rat, label='Hybrid-Rational (72 params)', color='red', linewidth=2)
plt.axvline(x=500, color='green', linestyle=':', label='Switch to LBFGS')

plt.title('PINN Benchmark: No Regularization (Pure Expressivity)')
plt.xlabel('Epochs')
plt.ylabel('Loss (Log Scale)')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.savefig("benchmark_comparison.png")
plt.show()