import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode 
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from timm import create_model
from timm.data.mixup import Mixup
from timm.data.auto_augment import rand_augment_transform
from timm.loss import SoftTargetCrossEntropy
from timm.scheduler import CosineLRScheduler

from tqdm import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed


# ============================================================
# Helper: Accuracy
# ============================================================
@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    bs = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / bs))
    return res


# ============================================================
# GELU-like Rational init (fitted on [-5,5])
# ============================================================
_GELU_INIT_P = torch.tensor([0.12768164, 0.93589880, 0.42351598, 0.06400502], dtype=torch.float32)
_GELU_INIT_Q = torch.tensor([0.20797062, 0.10903531], dtype=torch.float32)
_GELU_INIT_ALPHA = 1.1298931


# ============================================================
# RationalGroup1D: fp32 compute for stability
# out = x + alpha * (P(x)/(1+softplus(Q(x))) - x)
# ============================================================
class RationalGroup1D(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.p = nn.Parameter(torch.zeros(num_features, 4))   # p0..p3
        self.q = nn.Parameter(torch.zeros(num_features, 2))   # q1..q2
        init_alpha = float(os.environ.get("RAN_ALPHA_INIT", str(_GELU_INIT_ALPHA)))
        self.alpha = nn.Parameter(torch.full((num_features,), init_alpha))   # alpha

        with torch.no_grad():
            self.p.copy_(_GELU_INIT_P.unsqueeze(0).repeat(num_features, 1))
            self.q.copy_(_GELU_INIT_Q.unsqueeze(0).repeat(num_features, 1))
            self.alpha.fill_(init_alpha)

    def forward(self, x):
        dtype = x.dtype
        x_fp = x.float()

        x2 = x_fp * x_fp
        x3 = x2 * x_fp

        p0, p1, p2, p3 = self.p.float().unbind(dim=-1)
        num = p0 + p1 * x_fp + p2 * x2 + p3 * x3

        q1, q2 = self.q.float().unbind(dim=-1)
        q_part = q1 * x_fp + q2 * x2

        den = 1.0 + F.softplus(q_part)
        den = den.clamp_min(self.eps)

        r = num / den
        alpha = self.alpha.float()
        out = x_fp + alpha * (r - x_fp)

        return out.to(dtype)


# ============================================================
# MLP replacement (keep fc1/fc2 pretrained weights)
# ============================================================
class RationalMlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop=0.0, eps=1e-6):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = RationalGroup1D(hidden_features, eps=eps)
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def replace_vit_mlp_with_rational(model, eps=1e-6):
    for blk in model.blocks:
        old_mlp = blk.mlp
        in_features = old_mlp.fc1.in_features
        hidden_features = old_mlp.fc1.out_features
        out_features = old_mlp.fc2.out_features
        drop = getattr(old_mlp, "drop", 0.0) if hasattr(old_mlp, "drop") else 0.0

        new_mlp = RationalMlp(in_features, hidden_features, out_features, drop=drop, eps=eps)
        with torch.no_grad():
            new_mlp.fc1.weight.copy_(old_mlp.fc1.weight)
            new_mlp.fc1.bias.copy_(old_mlp.fc1.bias)
            new_mlp.fc2.weight.copy_(old_mlp.fc2.weight)
            new_mlp.fc2.bias.copy_(old_mlp.fc2.bias)
        blk.mlp = new_mlp
    return model


# ============================================================
# Data Module
# ============================================================
class ImageNetDataModule:
    def __init__(self, config):
        self.config = config
        self.train_dir = config["data"]["train_dir"]
        self.val_dir = config["data"]["val_dir"]
        self.batch_size = int(config["data"]["batch_size"])
        self.num_workers = int(config["data"]["num_workers"])

        self.mixup_cfg = config.get("mixup", {})
        self.setup_transforms()

        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.mixup_fn = None
        self.num_classes = None

    def setup_transforms(self):
      
        reprob = float(self.mixup_cfg.get("reprob", 0.25))
        ra = self.mixup_cfg.get("rand_augment", "rand-m9-mstd0.5-inc1")

        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            rand_augment_transform(
                config_str=ra,
                hparams={"translate_const": 117, "img_mean": (124, 116, 104)}
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=reprob),
        ])

        self.transform_val = transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def _align_val_class_to_idx(self, train_ds, val_ds):
        if val_ds.class_to_idx == train_ds.class_to_idx:
            return val_ds

        train_map = train_ds.class_to_idx
        new_samples = []
        for path, _ in val_ds.samples:
            cls = os.path.basename(os.path.dirname(path))
            if cls not in train_map:
                continue
            new_samples.append((path, train_map[cls]))

        if len(new_samples) == 0:
            raise RuntimeError("wrong val dataset after class_to_idx alignment")

        val_ds.samples = new_samples
        val_ds.targets = [s[1] for s in new_samples]
        val_ds.class_to_idx = train_map
        val_ds.classes = train_ds.classes
        return val_ds

    def setup(self):
        self.train_dataset = datasets.ImageFolder(self.train_dir, self.transform_train)
        self.val_dataset = datasets.ImageFolder(self.val_dir, self.transform_val)
        self.val_dataset = self._align_val_class_to_idx(self.train_dataset, self.val_dataset)
        self.num_classes = len(self.train_dataset.classes)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )

        # Mixup/Cutmix
        mixup_alpha = float(self.mixup_cfg.get("mixup_alpha", 0.8))
        cutmix_alpha = float(self.mixup_cfg.get("cutmix_alpha", 1.0))
        prob = float(self.mixup_cfg.get("mixup_prob", 1.0))
        switch_prob = float(self.mixup_cfg.get("switch_prob", 0.5))
        mode = self.mixup_cfg.get("mixup_mode", "batch")
        label_smoothing = float(self.mixup_cfg.get("label_smoothing", 0.1))

        self.mixup_fn = Mixup(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            prob=prob,
            switch_prob=switch_prob,
            mode=mode,
            label_smoothing=label_smoothing,
            num_classes=self.num_classes
        )


# ============================================================
# LLRD: layer id helper for ViT
# ============================================================
def get_vit_layer_id(name: str, num_blocks: int) -> int:
    # 0: embeddings
    if name.startswith("patch_embed") or name in ("cls_token", "pos_embed"):
        return 0
    # 1..num_blocks: blocks
    if name.startswith("blocks."):
        try:
            bid = int(name.split(".")[1])
            return bid + 1
        except Exception:
            return 0
    # last: head / norm
    return num_blocks + 1


def build_optimizer_llrd(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    layer_decay: float,
    head_lr_mult: float,
    rational_p_mult: float,
    rational_q_mult: float,
    rational_alpha_mult: float,
):
    no_decay_keywords = ("bias", "norm", "cls_token", "pos_embed")

    num_blocks = len(getattr(model, "blocks"))
    max_layer_id = num_blocks + 1

    param_groups = {}

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        layer_id = get_vit_layer_id(name, num_blocks)
        lr_scale = layer_decay ** (max_layer_id - layer_id)
        lr = base_lr * lr_scale

        # decide decay
        decay = True
        for kw in no_decay_keywords:
            if kw in name:
                decay = False
                break
        wd = weight_decay if decay else 0.0

        # tag multipliers
        mult = 1.0
        if "head" in name:
            mult = head_lr_mult
        elif ".mlp.act.p" in name:
            mult = rational_p_mult
            wd = 0.0
        elif ".mlp.act.q" in name:
            mult = rational_q_mult
            wd = 0.0
        elif ".mlp.act.alpha" in name:
            mult = rational_alpha_mult
            wd = 0.0

        group_lr = lr * mult
        group_name = f"layer{layer_id}_{'decay' if wd > 0 else 'nodecay'}_mult{mult:.3f}"

        if group_name not in param_groups:
            param_groups[group_name] = {"params": [], "lr": group_lr, "weight_decay": wd}
        param_groups[group_name]["params"].append(p)

    optimizer = optim.AdamW(list(param_groups.values()), betas=(0.9, 0.999))
    return optimizer


# ============================================================
# Trainer
# ============================================================
class RationalViTTrainer:
    def __init__(self, config, num_classes):
        self.config = config
        self.num_classes = num_classes

        mp = config.get("training", {}).get("mixed_precision", "fp16")
        self.accelerator = Accelerator(mixed_precision=mp)
        self.logger = get_logger(__name__)
        set_seed(int(config.get("seed", 42)))


        self.best_acc = 0.0
        self.global_step = 0

        self.setup_model()
        self.setup_criterion()

        self.optimizer = None
        self.scheduler = None
        self._sched_cfg = None

    def setup_model(self):

        self.model = create_model(
            "deit_tiny_patch16_224",
            pretrained=True,
            num_classes=self.num_classes
        )
        self.model = replace_vit_mlp_with_rational(self.model, eps=1e-6)

        if self.accelerator.is_main_process:
            self.logger.info("Model: pretrained DeiT-Tiny + Rational activation (fc1/fc2 preserved).")

        extra_path = self.config.get("training", {}).get("extra_pretrained_path", None)
        if extra_path and os.path.exists(extra_path):
            ckpt = torch.load(extra_path, map_location="cpu")
            state = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state, strict=False)
            if self.accelerator.is_main_process:
                self.logger.info(f"Loaded extra weights: {extra_path}")

    def setup_criterion(self):
        self.criterion_train = SoftTargetCrossEntropy()
        self.criterion_val = nn.CrossEntropyLoss()

    def setup_optimizer_scheduler(self, steps_per_epoch: int):
        tr = self.config["training"]
        base_lr = float(tr["learning_rate"])
        wd = float(tr["weight_decay"])
        epochs = int(tr["epochs"])


        layer_decay = float(tr.get("layer_decay", 0.75))          #  0.65~0.80
        head_lr_mult = float(tr.get("head_lr_mult", 1.0))         # head  1~2
        rational_p_mult = float(tr.get("rational_p_mult", 1.0))   # p 1~2
        rational_q_mult = float(tr.get("rational_q_mult", 0.5))   # q 0.3~0.8
        rational_alpha_mult = float(tr.get("rational_alpha_mult", 0.1))  # alpha 0.05~0.2

        self.optimizer = build_optimizer_llrd(
            self.model,
            base_lr=base_lr,
            weight_decay=wd,
            layer_decay=layer_decay,
            head_lr_mult=head_lr_mult,
            rational_p_mult=rational_p_mult,
            rational_q_mult=rational_q_mult,
            rational_alpha_mult=rational_alpha_mult,
        )

        # step-based cosine schedule
        min_lr = float(tr.get("min_lr", 1e-6))
        warmup_epochs = int(tr.get("warmup_epochs", 5))
        warmup_lr_init = float(tr.get("warmup_lr_init", base_lr * 0.01))

        total_steps = steps_per_epoch * epochs
        warmup_steps = steps_per_epoch * warmup_epochs

        self.scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=total_steps,
            lr_min=min_lr,
            warmup_lr_init=warmup_lr_init,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )

        if self.accelerator.is_main_process:
            self.logger.info(f"Optimizer: AdamW + LLRD layer_decay={layer_decay}")
            self.logger.info(f"LR mult: head={head_lr_mult}, p={rational_p_mult}, q={rational_q_mult}, alpha={rational_alpha_mult}")
            self.logger.info(f"Scheduler: cosine step-based total_steps={total_steps} warmup_steps={warmup_steps}")

    def prepare(self, datamodule: ImageNetDataModule):
        steps_per_epoch = len(datamodule.train_loader)
        self.setup_optimizer_scheduler(steps_per_epoch)

        self.model, self.optimizer, datamodule.train_loader, datamodule.val_loader = \
            self.accelerator.prepare(self.model, self.optimizer, datamodule.train_loader, datamodule.val_loader)

        self.train_loader = datamodule.train_loader
        self.val_loader = datamodule.val_loader
        self.mixup_fn = datamodule.mixup_fn

    def _get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total = 0
        correct1 = 0.0
        correct5 = 0.0

        log_interval = int(self.config.get("training", {}).get("log_interval", 200))
        grad_clip = float(self.config.get("training", {}).get("grad_clip", 1.0))

        pbar = tqdm(self.train_loader, disable=not self.accelerator.is_local_main_process)
        for step, (images, labels) in enumerate(pbar):
            images = images.to(self.accelerator.device, non_blocking=True)
            labels = labels.to(self.accelerator.device, non_blocking=True)

            if self.mixup_fn is not None:
                images, labels = self.mixup_fn(images, labels)
            labels_argmax = labels.argmax(dim=-1)

            self.optimizer.zero_grad(set_to_none=True)

            with self.accelerator.autocast():
                out = self.model(images)
                loss = self.criterion_train(out, labels)

            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.optimizer.step()

            self.global_step += 1
            self.scheduler.step_update(self.global_step)

            total_loss += loss.item()

            acc1, acc5 = accuracy(out, labels_argmax, topk=(1, 5))
            bs = labels.size(0)
            total += bs
            correct1 += acc1.item() * bs / 100.0
            correct5 += acc5.item() * bs / 100.0

            if self.accelerator.is_main_process and step % log_interval == 0 and step > 0:
                pbar.set_description(f"Ep{epoch+1} Step{step}/{len(self.train_loader)} Loss{loss.item():.3f} LR{self._get_lr():.6f}")

        loss_t = torch.tensor(total_loss, device=self.accelerator.device)
        total_t = torch.tensor(total, device=self.accelerator.device)
        c1_t = torch.tensor(correct1, device=self.accelerator.device)
        c5_t = torch.tensor(correct5, device=self.accelerator.device)

        loss_sum = self.accelerator.gather(loss_t).sum().item()
        total_sum = self.accelerator.gather(total_t).sum().item()
        c1_sum = self.accelerator.gather(c1_t).sum().item()
        c5_sum = self.accelerator.gather(c5_t).sum().item()

        steps_total = len(self.train_loader) * self.accelerator.num_processes
        train_loss = loss_sum / steps_total
        train_acc1 = 100.0 * c1_sum / total_sum
        train_acc5 = 100.0 * c5_sum / total_sum
        return train_loss, train_acc1, train_acc5

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        total = 0
        correct1 = 0.0
        correct5 = 0.0

        pbar = tqdm(self.val_loader, disable=not self.accelerator.is_local_main_process)
        for images, labels in pbar:
            images = images.to(self.accelerator.device, non_blocking=True)
            labels = labels.to(self.accelerator.device, non_blocking=True)

            with self.accelerator.autocast():
                out = self.model(images)

            loss = self.criterion_val(out.float(), labels)
            total_loss += loss.item()

            acc1, acc5 = accuracy(out, labels, topk=(1, 5))
            bs = labels.size(0)
            total += bs
            correct1 += acc1.item() * bs / 100.0
            correct5 += acc5.item() * bs / 100.0

        loss_t = torch.tensor(total_loss, device=self.accelerator.device)
        total_t = torch.tensor(total, device=self.accelerator.device)
        c1_t = torch.tensor(correct1, device=self.accelerator.device)
        c5_t = torch.tensor(correct5, device=self.accelerator.device)

        loss_sum = self.accelerator.gather(loss_t).sum().item()
        total_sum = self.accelerator.gather(total_t).sum().item()
        c1_sum = self.accelerator.gather(c1_t).sum().item()
        c5_sum = self.accelerator.gather(c5_t).sum().item()

        steps_total = len(self.val_loader) * self.accelerator.num_processes
        val_loss = loss_sum / steps_total
        val_acc1 = 100.0 * c1_sum / total_sum
        val_acc5 = 100.0 * c5_sum / total_sum
        return val_loss, val_acc1, val_acc5

    def save_checkpoint(self, epoch, val_acc):
        ckpt_path = self.config["training"]["checkpoint_path"].split('.')[0] + '_deit.pth'
        if self.accelerator.is_main_process and val_acc > self.best_acc:
            self.best_acc = val_acc
            state = {
                "epoch": epoch,
                "model_state_dict": self.accelerator.get_state_dict(self.model),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_acc": self.best_acc,
                "global_step": self.global_step,
            }
            self.accelerator.save(state, ckpt_path)
            self.logger.info(f"Saved best: epoch={epoch+1} acc={val_acc:.2f}% -> {ckpt_path}")

    def train(self):
        epochs = int(self.config["training"]["epochs"])

        if self.accelerator.is_main_process:
            with open("training_log_deit.txt", "w") as f:
                f.write("Epoch,Train_Loss,Train_Acc1,Train_Acc5,Val_Loss,Val_Acc1,Val_Acc5,LR\n")

        for epoch in range(epochs):
            if self.accelerator.is_main_process:
                self.logger.info(f"\nEpoch {epoch+1}/{epochs}")

            train_loss, train_acc1, train_acc5 = self.train_one_epoch(epoch)
            val_loss, val_acc1, val_acc5 = self.validate()

            lr_now = self._get_lr()
            if val_acc1 > self.best_acc:
                self.best_acc = val_acc1

            if self.accelerator.is_main_process:
                self.logger.info(f"Train: loss={train_loss:.4f} acc1={train_acc1:.2f} acc5={train_acc5:.2f}")
                self.logger.info(f"Val:   loss={val_loss:.4f} acc1={val_acc1:.2f} acc5={val_acc5:.2f}")
                self.logger.info(f"LR:    {lr_now:.6f}")

                with open("training_log.txt", "a") as f:
                    f.write(f"{epoch+1},{train_loss:.4f},{train_acc1:.2f},{train_acc5:.2f},"
                            f"{val_loss:.4f},{val_acc1:.2f},{val_acc5:.2f},{lr_now:.6f}\n")

            self.save_checkpoint(epoch, val_acc1)


def main():
    config_path = os.path.join(os.path.dirname(__file__), "configs", "config_for_7.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config_for_7.yaml not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Gate init (alpha) can be controlled via config; used by RationalGroup1D
    os.environ["RAN_ALPHA_INIT"] = str(config.get("training", {}).get("rational_alpha_init", _GELU_INIT_ALPHA))

    set_seed(int(config.get("seed", 42)))

    datamodule = ImageNetDataModule(config)
    datamodule.setup()

    trainer = RationalViTTrainer(config, num_classes=datamodule.num_classes)
    trainer.prepare(datamodule)

    if trainer.accelerator.is_main_process:
        trainer.logger.info(f"Dataset: train={len(datamodule.train_dataset)} val={len(datamodule.val_dataset)} classes={datamodule.num_classes}")

    trainer.train()


if __name__ == "__main__":
    main()