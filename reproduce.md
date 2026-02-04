# Reproducing Rational-ANOVA Networks (RAN)

This repo contains the code used in the paper **Rational ANOVA Networks**.

## 1) Vision benchmarks (Table 1)
Entry: `run_MINIST_FMNIST_CIFAR-10_EMNIST-Let_SVH_CIFAR-100.py`

Example:
```bash
python run_MINIST_FMNIST_CIFAR-10_EMNIST-Let_SVH_CIFAR-100.py --dataset CIFAR10 --params 1000000 --seed 42
```

## 2) ViT on ImageNet-1K (Table 2)
Entry: `vit_ran.py`

1) Edit `configs/config_for_7.yaml` and set ImageNet paths.
2) Run:
```bash
python vit_ran.py
```

## 3) PolyU real-world denoising (Figure 4)
Entry: `runPolyU.py`

Set `DATA_ROOT` inside the script (paired `Real/` and `Mean/` folders), then:
```bash
python runPolyU.py
```

## 4) TabArena efficiency (Figure 5)
Entry: `TAB_run.py`

```bash
python TAB_run.py
```

## 5) Lorentzian potential (Supplementary, Appendix B)
Entry: `pinn.py`

```bash
python pinn.py
```
