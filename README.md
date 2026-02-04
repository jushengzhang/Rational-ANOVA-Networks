# Rational ANOVA Networks (RAN)

This repository contains the reference implementation for **Rational ANOVA Networks (RAN)**.
RAN combines (i) a **Functional ANOVA** topology (main effects + sparse pairwise interactions) with
(ii) **learnable Padé-style rational units** with **strictly positive denominators** and **residual gating**
for stable deep training.

## Quick start

```bash
pip install -r requirements.txt
```

## Paper ↔ Code mapping (high-level)

- **Rational 1D / 2D units + positive denominator** (Eq. 4–5) → `run_MINIST_*.py`, `vit_ran.py`
- **Residual gating (identity-safe)** (Eq. 6–7) → gate `alpha` in the rational units (default init near 0)
- **ANOVA topology + sparse interaction set S** (Eq. 2–3) → `DeepRationalANOVA` in `run_MINIST_*.py`
- **Budget matching / dynamic |S|** (Eq. 10) → `estimate_K()` in `run_MINIST_*.py`
- **Drop-in ViT FFN replacement** (Eq. 8) → `replace_vit_mlp_with_rational()` in `vit_ran.py`

For step-by-step reproduction commands, see **`reproduce.md`**.

## Main entry points

- **Vision benchmarks (Table 1):** `run_MINIST_FMNIST_CIFAR-10_EMNIST-Let_SVH_CIFAR-100.py`
- **ViT (Table 2):** `vit_ran.py` + `configs/config_for_7.yaml`
- **PolyU denoising (Fig. 4):** `runPolyU.py`
- **TabArena (Fig. 5):** `TAB_run.py`
- **Lorentzian potential (Supp. B):** `pinn.py`

## Notes

- `configs/config_for_7.yaml` is a **template**. You must set ImageNet paths before running.
- The ViT script uses `timm` + `accelerate` and logs to `training_log_deit.txt`.
