import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import itertools
from autogluon.tabular.models import AbstractModel
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ==========================================
# 1. Basic Component: 1D Rational Layer
# ==========================================
class RationalLayer1D_Dynamic(nn.Module):
    def __init__(self, in_features, out_features, degree_P=3, degree_Q=2, num_heads=1):
        super().__init__()
        self.degree_P = degree_P
        self.degree_Q = degree_Q
        self.num_heads = num_heads
        self.terms_P = degree_P + 1
        self.terms_Q = degree_Q + 1
        self.p_coeffs = nn.Parameter(torch.randn(out_features, num_heads, in_features, self.terms_P) * 0.01)
        self.q_coeffs = nn.Parameter(torch.randn(out_features, num_heads, in_features, self.terms_Q) * 0.01)
        with torch.no_grad():
            self.q_coeffs[:, :, :, 0] = 1.0 
            self.p_coeffs[:, :, :, 0] += 0.1

    def forward(self, x):
        powers_P = torch.arange(self.degree_P + 1, device=x.device).float()
        powers_Q = torch.arange(self.degree_Q + 1, device=x.device).float()
        basis_P = x.unsqueeze(1).unsqueeze(1).unsqueeze(-1).pow(powers_P)
        basis_Q = x.unsqueeze(1).unsqueeze(1).unsqueeze(-1).pow(powers_Q)
        P = (basis_P * self.p_coeffs.unsqueeze(0)).sum(dim=-1)
        Q = (basis_Q * self.q_coeffs.unsqueeze(0)).sum(dim=-1)
        out_per_head = P / (torch.abs(Q) + 1e-5)
        return out_per_head.sum(dim=2).sum(dim=-1)

# ==========================================
# 2. Basic Component: 2D Interaction Layer (Memory Optimized)
# ==========================================
class RationalLayer2D_Dynamic(nn.Module):
    """
    Memory Optimized 2D Interaction Layer.
    Uses 'einsum' instead of broadcasting to significantly reduce VRAM usage.
    """
    def __init__(self, in_features, out_features, degree_P=3, degree_Q=2, num_heads=1):
        super().__init__()
        # Strategy: If there are too many features, limit interaction pairs to prevent VRAM explosion
        self.max_features = 50 
        
        if in_features > self.max_features:
            self.feature_idx = torch.arange(self.max_features)
            real_in = self.max_features
        else:
            self.feature_idx = torch.arange(in_features)
            real_in = in_features
            
        self.pairs = list(itertools.combinations(range(real_in), 2))
        self.num_pairs = len(self.pairs)
        self.num_heads = num_heads
        
        if self.num_pairs > 0:
            self.register_buffer('pow_P', self._generate_powers(degree_P))
            self.register_buffer('pow_Q', self._generate_powers(degree_Q))
            
            terms_P = len(self.pow_P)
            terms_Q = len(self.pow_Q)
            
            # [out, heads, pairs, terms]
            self.p_coeffs = nn.Parameter(torch.randn(out_features, num_heads, self.num_pairs, terms_P) * 0.01)
            self.q_coeffs = nn.Parameter(torch.randn(out_features, num_heads, self.num_pairs, terms_Q) * 0.01)
            
            with torch.no_grad():
                self.q_coeffs[:, :, :, 0] = 1.0

    def _generate_powers(self, degree):
        powers = []
        for d in range(degree + 1):
            for i in range(d + 1):
                j = d - i
                powers.append([i, j])
        return torch.tensor(powers, dtype=torch.float32)

    def forward(self, x):
        if self.num_pairs == 0:
            return 0.0
        
        # 1. Feature Truncation (Prevent dimension explosion)
        if x.shape[1] > len(self.feature_idx):
            x = x[:, self.feature_idx]

        # 2. Prepare Data [Batch, Pairs, 1]
        idx_1 = [p[0] for p in self.pairs]
        idx_2 = [p[1] for p in self.pairs]
        x1 = x[:, idx_1].unsqueeze(-1)
        x2 = x[:, idx_2].unsqueeze(-1)

        # 3. Memory Optimization Core: Stepwise Calculation + Einsum
        basis_P = (x1 ** self.pow_P[:, 0]) * (x2 ** self.pow_P[:, 1]) 
        basis_Q = (x1 ** self.pow_Q[:, 0]) * (x2 ** self.pow_Q[:, 1])

        # Use einsum for polynomial multiplication 'bpt, ohpt -> bohp'
        P = torch.einsum('bpt, ohpt -> bohp', basis_P, self.p_coeffs)
        Q = torch.einsum('bpt, ohpt -> bohp', basis_Q, self.q_coeffs)
        
        out = P / (torch.abs(Q) + 1e-5)
        
        return out.sum(dim=-1).sum(dim=-1)

# ==========================================
# 3. Network Assembly Layer 
# ==========================================
class RationalANOVALayer(nn.Module):
    def __init__(self, in_features, out_features, degree_P=3, degree_Q=2, num_heads=1):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.main_effects = RationalLayer1D_Dynamic(in_features, out_features, degree_P, degree_Q, num_heads)
        self.use_interactions = in_features > 1
        if self.use_interactions:
            self.interactions = RationalLayer2D_Dynamic(in_features, out_features, degree_P, degree_Q, num_heads)
    def forward(self, x):
        y = self.bias + self.main_effects(x)
        if self.use_interactions:
            y = y + self.interactions(x)
        return y

class ThreeLayerRationalNet(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, degree_P=3, degree_Q=2, num_heads=1):
        super().__init__()
        self.layer1 = RationalANOVALayer(input_dim, hidden_dim1, degree_P, degree_Q, num_heads)
        self.bn1 = nn.LayerNorm(hidden_dim1)
        self.layer2 = RationalANOVALayer(hidden_dim1, hidden_dim2, degree_P, degree_Q, num_heads)
        self.bn2 = nn.LayerNorm(hidden_dim2)
        self.layer3 = RationalANOVALayer(hidden_dim2, output_dim, degree_P, degree_Q, num_heads)
    def forward(self, x):
        x = self.bn1(self.layer1(x))
        x = self.bn2(self.layer2(x))
        return self.layer3(x)

# ==========================================
# 4. AutoGluon Adapter (Fixed Version)
# ==========================================
class RationalTABModel(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self.scaler = None
        self.y_scaler = None
        self.label_encoder = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _to_numpy(self, data):
        """Utility: Safely convert Pandas/List to Numpy"""
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.values
        if isinstance(data, list):
            return np.array(data)
        return data  # Assume it is already a numpy array

    def _preprocess(self, X):
        X = X.copy()
        # If DataFrame, handle categories; if Numpy, assume already numerical
        if isinstance(X, pd.DataFrame):
            for col in X.select_dtypes(include=['object', 'category']).columns:
                X[col] = X[col].astype('category').cat.codes
            X = X.fillna(0)
            X_np = X.values
        else:
            X_np = X
            
        # Ensure float32 again
        X_np = np.nan_to_num(X_np.astype(np.float32))

        if self.scaler is None:
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(X_np)
        return self.scaler.transform(X_np)

    def _fit(self, X, y, **kwargs):
        try:
            epochs = self.params.get('epochs', 30)
            lr = self.params.get('lr', 0.01)
            batch_size = self.params.get('batch_size', 256)
            
            # 1. Feature Preprocessing
            X_np = self._preprocess(X)
            
            # 2. Label Preprocessing (Core fix: Convert to numpy first)
            y_np = self._to_numpy(y)
            out_dim = 1
            
            # === Classification Task Handling ===
            if self.problem_type in ['binary', 'multiclass']:
                self.label_encoder = LabelEncoder()
                y_encoded = self.label_encoder.fit_transform(y_np.ravel()) # ravel() to prevent dimension mismatch
                out_dim = len(self.label_encoder.classes_)
                
                if out_dim == 2: 
                    out_dim = 1
                    y_t = torch.tensor(y_encoded, dtype=torch.float32).to(self.device)
                else:
                    y_t = torch.tensor(y_encoded, dtype=torch.long).to(self.device)
                    
            # === Regression Task Handling ===
            elif self.problem_type == 'regression':
                self.y_scaler = StandardScaler()
                # Fix: Ensure reshape operation is performed on numpy array
                y_reshaped = y_np.reshape(-1, 1)
                y_scaled = self.y_scaler.fit_transform(y_reshaped).flatten()
                out_dim = 1
                y_t = torch.tensor(y_scaled, dtype=torch.float32).to(self.device)

            X_t = torch.tensor(X_np, dtype=torch.float32).to(self.device)
            
            # 3. Prepare Data
            dataset = torch.utils.data.TensorDataset(X_t, y_t)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # 4. Initialize Model
            # Note: This assumes ThreeLayerRationalNet is defined in this file or imported
            # from rational_model import ThreeLayerRationalNet 
            
            self.model = ThreeLayerRationalNet(
                input_dim=X_t.shape[1],
                hidden_dim1=self.params.get('hidden_dim1', 32),
                hidden_dim2=self.params.get('hidden_dim2', 16),
                output_dim=out_dim,
                degree_P=self.params.get('degree_P', 3),
                degree_Q=self.params.get('degree_Q', 2),
                num_heads=self.params.get('num_heads', 2)
            ).to(self.device)
            
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
            
            if self.problem_type == 'regression':
                criterion = nn.MSELoss()
            elif out_dim == 1:
                criterion = nn.BCEWithLogitsLoss()
            else:
                criterion = nn.CrossEntropyLoss()
            
            # 5. Training
            self.model.train()
            for epoch in range(epochs):
                for bx, by in loader:
                    optimizer.zero_grad()
                    pred = self.model(bx)
                    loss = criterion(pred.squeeze(), by)
                    
                    if torch.isnan(loss):
                        print("⚠️ Warning: Loss is NaN, stopping training early.")
                        break
                        
                    loss.backward()
                    optimizer.step()
        except Exception as e:
            # Print detailed error stack for debugging
            import traceback
            traceback.print_exc()
            raise e

    def _predict_proba(self, X, **kwargs):
        X_np = self._preprocess(X)
        X_t = torch.tensor(X_np, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t)
            
            if self.problem_type == 'regression':
                pred_scaled = logits.cpu().numpy().flatten()
                if self.y_scaler:
                    return self.y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
                return pred_scaled

            elif self.problem_type == 'multiclass':
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                return pd.DataFrame(data=probs, columns=self.label_encoder.classes_, index=X.index if hasattr(X, 'index') else None)
            
            else: # binary
                prob1 = torch.sigmoid(logits).cpu().numpy().flatten()
                prob0 = 1.0 - prob1
                return pd.DataFrame(
                    data=np.column_stack([prob0, prob1]), 
                    columns=self.label_encoder.classes_, 
                    index=X.index if hasattr(X, 'index') else None
                )

    def predict(self, X, **kwargs):
        if self.problem_type == 'regression':
            return self._predict_proba(X, **kwargs)
        else:
            return super().predict(X, **kwargs)

    def predict_from_proba(self, y_pred_proba):
        if isinstance(y_pred_proba, pd.DataFrame):
            y_pred_proba = y_pred_proba.values
        return super().predict_from_proba(y_pred_proba)