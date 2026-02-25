"""
糖尿病视网膜病变分类 - APTOS 2019
模型: EfficientNet-V2 (timm)
框架: PyTorch

数据集目录结构：
    aptos2019/
    ├── train_images/
    ├── val_images/
    ├── test_images/
    ├── train.csv       (id_code, diagnosis)
    ├── valid.csv         (id_code, diagnosis)
    └── test.csv        (id_code, diagnosis)

分级标准 (0-4):
    0 - No DR
    1 - Mild
    2 - Moderate
    3 - Severe
    4 - Proliferative DR
"""

import os
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as T
import timm


# ─────────────────────────────────────────
#  配置
# ─────────────────────────────────────────
class Config:
    # 数据路径（根据实际情况修改）
    DATA_DIR      = "D:/mygithub/data/"
    TRAIN_CSV     = os.path.join(DATA_DIR, "train.csv")
    VAL_CSV       = os.path.join(DATA_DIR, "valid.csv")
    TEST_CSV      = os.path.join(DATA_DIR, "test.csv")
    TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
    VAL_IMG_DIR   = os.path.join(DATA_DIR, "val_images")
    TEST_IMG_DIR  = os.path.join(DATA_DIR, "test_images")

    # 模型
    MODEL_NAME     = "tf_efficientnetv2_s"
    NUM_CLASSES    = 5
    PRETRAINED     = False
    LOCAL_WEIGHTS  = r"D:\mygithub\data\model.safetensors"  # 本地权重路径

    # 训练
    SEED          = 42
    EPOCHS        = 50
    BATCH_SIZE    = 16
    NUM_WORKERS   = 4
    IMG_SIZE      = 384
    LR            = 1e-4
    WEIGHT_DECAY  = 1e-5
    EARLY_STOP    = 7

    # 混合精度
    AMP           = True

    # 保存
    SAVE_DIR      = "./checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)
    BEST_CKPT     = os.path.join(SAVE_DIR, "best_model.pth")

    DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(cfg.SEED)

#  数据集
class APTOSDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, str(row["id_code"]) + ".png")
        image    = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(int(row["diagnosis"]), dtype=torch.long)
        return image, label


#  数据增强
def get_transforms(phase: str):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if phase == "train":
        return T.Compose([
            T.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(20),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    else:   # val / test
        return T.Compose([
            T.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])


#  模型
class EfficientNetV2Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            cfg.MODEL_NAME,
            pretrained=False,
            num_classes=0,
            global_pool="avg"
        )
        # 从本地加载预训练权重
        from safetensors.torch import load_file
        state_dict = load_file(cfg.LOCAL_WEIGHTS)
        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
        print(f"  [权重加载] missing={len(missing)}  unexpected={len(unexpected)}")
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, cfg.NUM_CLASSES),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


#  标签平滑损失
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        n_cls     = logits.size(1)
        log_probs = nn.functional.log_softmax(logits, dim=1)
        nll       = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        smooth    = -log_probs.mean(dim=1)
        return ((1 - self.smoothing) * nll + self.smoothing * smooth).mean()


#  类别权重（应对样本不均衡）
def compute_class_weights(df, num_classes=5, device="cpu"):
    counts  = df["diagnosis"].value_counts().sort_index()
    counts  = counts.reindex(range(num_classes), fill_value=1)
    weights = 1.0 / counts.values.astype(float)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32).to(device)


#  训练 & 验证函数
def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(cfg.DEVICE, non_blocking=True)
        labels = labels.to(cfg.DEVICE, non_blocking=True)

        optimizer.zero_grad()
        with autocast(enabled=cfg.AMP):
            logits = model(images)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(cfg.DEVICE, non_blocking=True)
        labels = labels.to(cfg.DEVICE, non_blocking=True)

        with autocast(enabled=cfg.AMP):
            logits = model(images)
            loss   = criterion(logits, labels)

        preds = logits.argmax(1)
        total_loss += loss.item() * labels.size(0)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    return total_loss / total, correct / total, kappa, all_preds, all_labels


#  主流程
def main():
    print(f"{'='*55}")
    print(f"  糖尿病视网膜病变分类  |  {cfg.MODEL_NAME}")
    print(f"  Device: {cfg.DEVICE}")
    print(f"{'='*55}\n")

    # ── 读取 CSV ──────────────────────────
    train_df = pd.read_csv(cfg.TRAIN_CSV)
    val_df   = pd.read_csv(cfg.VAL_CSV)
    test_df  = pd.read_csv(cfg.TEST_CSV)

    print("数据集大小:")
    print(f"  Train : {len(train_df)}")
    print(f"  Val   : {len(val_df)}")
    print(f"  Test  : {len(test_df)}")

    print("\n训练集各类别分布:")
    print(train_df["diagnosis"].value_counts().sort_index().to_string())
    print("\n验证集各类别分布:")
    print(val_df["diagnosis"].value_counts().sort_index().to_string())

    # ── 数据加载 ──────────────────────────
    train_ds = APTOSDataset(train_df, cfg.TRAIN_IMG_DIR, get_transforms("train"))
    val_ds   = APTOSDataset(val_df,   cfg.VAL_IMG_DIR,   get_transforms("val"))
    test_ds  = APTOSDataset(test_df,  cfg.TEST_IMG_DIR,  get_transforms("val"))

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                              shuffle=True,  num_workers=cfg.NUM_WORKERS,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE,
                              shuffle=False, num_workers=cfg.NUM_WORKERS,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE,
                              shuffle=False, num_workers=cfg.NUM_WORKERS,
                              pin_memory=True)

    # ── 模型 ──────────────────────────────
    model = EfficientNetV2Classifier().to(cfg.DEVICE)
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f} M\n")

    # ── 损失函数 ──────────────────────────
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

    # ── 优化器：分层学习率（backbone 用更小 lr）─
    backbone_params   = {"params": model.backbone.parameters(),    "lr": cfg.LR * 0.1}
    classifier_params = {"params": model.classifier.parameters(),  "lr": cfg.LR}
    optimizer = optim.AdamW([backbone_params, classifier_params], weight_decay=cfg.WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-6
    )
    scaler = GradScaler(enabled=cfg.AMP)

    # ── 训练循环 ──────────────────────────
    best_kappa = -1.0
    no_improve = 0
    history    = []

    print(f"{'Epoch':>6} | {'TrainLoss':>10} | {'TrainAcc':>9} | {'ValLoss':>8} | {'ValAcc':>7} | {'Kappa':>7} | {'LR':>9}")
    print("-" * 75)

    for epoch in range(1, cfg.EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
        vl_loss, vl_acc, kappa, preds, labels = evaluate(model, val_loader, criterion)
        scheduler.step()
        lr = optimizer.param_groups[1]["lr"]

        history.append({
            "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": vl_loss, "val_acc": vl_acc, "kappa": kappa
        })

        marker = ""
        if kappa > best_kappa:
            best_kappa = kappa
            no_improve = 0
            torch.save({
                "epoch":             epoch,
                "model_state_dict":  model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_kappa":        best_kappa,
            }, cfg.BEST_CKPT)
            marker = "  ✓ best"
        else:
            no_improve += 1

        print(f"{epoch:>6} | {tr_loss:>10.4f} | {tr_acc:>9.4f} | "
              f"{vl_loss:>8.4f} | {vl_acc:>7.4f} | {kappa:>7.4f} | {lr:>9.2e}{marker}")

        if no_improve >= cfg.EARLY_STOP:
            print(f"\n Early stopping at epoch {epoch} (no improvement for {cfg.EARLY_STOP} epochs)")
            break

    # ── 保存训练历史 ──────────────────────
    pd.DataFrame(history).to_csv("train_history.csv", index=False)
    print(f"\n训练历史已保存至train_history.csv")

    # ── 在验证集上最终评估 ────────────────
    print(f"\n{'='*55}")
    print("  加载最佳模型，在验证集上最终评估")
    print(f"{'='*55}")
    ckpt = torch.load(cfg.BEST_CKPT, map_location=cfg.DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Best checkpoint: epoch={ckpt['epoch']}, kappa={ckpt['best_kappa']:.4f}")

    _, val_acc, val_kappa, val_preds, val_labels = evaluate(model, val_loader, criterion)
    class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    print(f"\nVal Accuracy : {val_acc:.4f}")
    print(f"Val Kappa    : {val_kappa:.4f}")
    print("\nClassification Report (Val):")
    print(classification_report(val_labels, val_preds, target_names=class_names))
    print("Confusion Matrix (Val):")
    cm = confusion_matrix(val_labels, val_preds)
    print(pd.DataFrame(cm, index=class_names, columns=class_names).to_string())

    # ── 在测试集上预测 ────────────────────
    print(f"\n{'='*55}")
    print("  在测试集上预测")
    print(f"{'='*55}")
    _, test_acc, test_kappa, test_preds, test_labels = evaluate(model, test_loader, criterion)
    print(f"Test Accuracy : {test_acc:.4f}")
    print(f"Test Kappa    : {test_kappa:.4f}")
    print("\nClassification Report (Test):")
    print(classification_report(test_labels, test_preds, target_names=class_names))

    result_df = test_df.copy()
    result_df["prediction"] = test_preds
    result_df.to_csv("test_predictions.csv", index=False)
    print("\n测试集预测结果已保存至test_predictions.csv")


if __name__ == "__main__":
    main()