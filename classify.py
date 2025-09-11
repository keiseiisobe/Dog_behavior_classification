import argparse
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt


# ------------------------------
# Reproducibility helpers
# ------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------
# Constants and Label Space
# ------------------------------
BEHAVIOR_CLASSES: List[str] = [
    "Lying chest",
    "Sniffing",
    "Playing",
    "Panting",
    "Walking",
    "Trotting",
    "Sitting",
    "Standing",
    "Eating",
    "Pacing",
    "Drinking",
    "Shaking",
    "Carrying object",
    "Tugging",
    "Galloping",
    "Jumping",
    "Bowing",
]

LABEL_TO_INDEX: Dict[str, int] = {label: idx for idx, label in enumerate(BEHAVIOR_CLASSES)}
INDEX_TO_LABEL: Dict[int, str] = {idx: label for label, idx in LABEL_TO_INDEX.items()}


# ------------------------------
# Dataset
# ------------------------------
@dataclass
class WindowingConfig:
    window_size: int = 64
    stride: int = 32
    require_pure_label: bool = False  # if True, drop windows where labels are mixed


class DogBehaviorSequenceDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        feature_columns: Optional[List[str]] = None,
        label_column: str = "Behavior_1",
        group_column: str = "DogID",
        windowing: Optional[WindowingConfig] = None,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        if feature_columns is None:
            feature_columns = [
                "ANeck_x",
                "ANeck_y",
                "ANeck_z",
                "GNeck_x",
                "GNeck_y",
                "GNeck_z",
            ]

        self.csv_path = csv_path
        self.feature_columns = feature_columns
        self.label_column = label_column
        self.group_column = group_column
        self.windowing = windowing or WindowingConfig()
        self.normalize = normalize

        df = pd.read_csv(csv_path)
        # Filter to supported labels only
        df = df[df[self.label_column].isin(BEHAVIOR_CLASSES)].copy()

        # Sort within groups to retain temporal order if index is not ordered
        if self.group_column in df.columns:
            df = df.sort_values([self.group_column]).reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        # Group by dog to generate windows within each dog's sequence
        groups: List[Tuple[np.ndarray, np.ndarray]] = []
        if self.group_column in df.columns:
            for _, sub in df.groupby(self.group_column):
                x = sub[self.feature_columns].values.astype(np.float32)
                y = sub[self.label_column].map(LABEL_TO_INDEX).values.astype(np.int64)
                groups.append((x, y))
        else:
            x = df[self.feature_columns].values.astype(np.float32)
            y = df[self.label_column].map(LABEL_TO_INDEX).values.astype(np.int64)
            groups.append((x, y))

        # Compute normalization stats from this split (train/val/test each normalized independently by default)
        # Optionally, users could pass in precomputed stats if needed.
        if self.normalize:
            all_x = np.concatenate([g[0] for g in groups], axis=0)
            self.feature_mean = all_x.mean(axis=0, keepdims=True)
            self.feature_std = all_x.std(axis=0, keepdims=True) + 1e-6
        else:
            self.feature_mean = np.zeros((1, len(self.feature_columns)), dtype=np.float32)
            self.feature_std = np.ones((1, len(self.feature_columns)), dtype=np.float32)

        # Window the sequences
        self.windows: List[Tuple[np.ndarray, int]] = []
        W = self.windowing.window_size
        S = self.windowing.stride
        for x, y in groups:
            if self.normalize:
                x = (x - self.feature_mean) / self.feature_std
            n = len(x)
            if n < W:
                # pad short sequences by repeating last frame
                pad_len = W - n
                x_pad = np.concatenate([x, np.repeat(x[-1:, :], pad_len, axis=0)], axis=0)
                y_pad = np.concatenate([y, np.repeat(y[-1:], pad_len, axis=0)], axis=0)
                label = self._aggregate_labels(y_pad)
                if (not self.windowing.require_pure_label) or self._is_pure(y_pad):
                    self.windows.append((x_pad, label))
                continue

            for start in range(0, n - W + 1, S):
                end = start + W
                x_w = x[start:end]
                y_w = y[start:end]
                if self.windowing.require_pure_label and not self._is_pure(y_w):
                    continue
                label = self._aggregate_labels(y_w)
                self.windows.append((x_w, label))

        if len(self.windows) == 0:
            raise ValueError(f"No windows produced from {csv_path}. Check labels and window size.")

    @staticmethod
    def _is_pure(y_window: np.ndarray) -> bool:
        return np.all(y_window == y_window[0])

    @staticmethod
    def _aggregate_labels(y_window: np.ndarray) -> int:
        # Majority label
        vals, counts = np.unique(y_window, return_counts=True)
        return int(vals[np.argmax(counts)])

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.windows[idx]
        x_t = torch.from_numpy(x)  # (T, C)
        y_t = torch.tensor(y, dtype=torch.long)
        return x_t, y_t


# ------------------------------
# Model: Transformer Encoder for sequence classification
# ------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        T = x.size(1)
        return x + self.pe[:, :T]


class TransformerTimeSeriesClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        use_cls_token: bool = True,
    ) -> None:
        super().__init__()
        self.use_cls_token = use_cls_token
        self.input_proj = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        cls_dim = d_model
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if use_cls_token else None
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(cls_dim, cls_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cls_dim, num_classes),
        )

        self._init_parameters()

    def _init_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        h = self.input_proj(x)  # (B, T, d_model)
        if self.use_cls_token:
            B = h.size(0)
            cls_tok = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
            h = torch.cat([cls_tok, h], dim=1)  # (B, 1+T, d_model)
        h = self.positional_encoding(h)
        h = self.encoder(h)  # (B, 1+T, d_model) or (B, T, d_model)

        if self.use_cls_token:
            h_cls = h[:, 0]
        else:
            h_cls = h.mean(dim=1)
        h_cls = self.norm(h_cls)
        logits = self.head(h_cls)
        return logits


# ------------------------------
# Training / Evaluation utilities
# ------------------------------
def save_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    out_path: str,
    normalize: bool = True,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    if normalize:
        cm = cm.astype(np.float32)
        row_sums = cm.sum(axis=1, keepdims=True) + 1e-9
        cm = cm / row_sums

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix" + (" (normalized)" if normalize else ""),
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            text = f"{val:.2f}" if normalize else str(int(val))
            ax.text(j, i, text, ha="center", va="center", color="white" if val > thresh else "black")
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, scaler: Optional[torch.cuda.amp.GradScaler], criterion: nn.Module, grad_clip: Optional[float] = 1.0) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(x)
            loss = criterion(logits, y)
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += x.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: Optional[nn.Module] = None) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        if criterion is not None:
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += x.size(0)
    avg_loss = total_loss / total_samples if criterion is not None else 0.0
    acc = total_correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, acc


@torch.no_grad()
def predict_all(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[List[int], List[int]]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().tolist()
        y_true.extend(y.tolist())
        y_pred.extend(preds)
    return y_true, y_pred


# ------------------------------
# CLI and Main
# ------------------------------
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transformer-based Dog Behavior Classification (PyTorch)")
    parser.add_argument("--train_csv", type=str, default="datasets/DogMoveData_train.csv")
    parser.add_argument("--val_csv", type=str, default="datasets/DogMoveData_val.csv")
    parser.add_argument("--test_csv", type=str, default="datasets/DogMoveData_test.csv")
    parser.add_argument("--output_dir", type=str, default="outputs")

    # Model
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dim_ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_cls_token", action="store_true")

    # Data
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--require_pure_label", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)

    # Train
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Run mode
    parser.add_argument("--test_only", action="store_true", help="Skip training and only evaluate on test")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)

    windowing = WindowingConfig(
        window_size=args.window_size,
        stride=args.stride,
        require_pure_label=args.require_pure_label,
    )

    # Datasets
    train_ds = DogBehaviorSequenceDataset(args.train_csv, windowing=windowing, normalize=True)
    val_ds = DogBehaviorSequenceDataset(args.val_csv, windowing=windowing, normalize=True)
    test_ds = DogBehaviorSequenceDataset(args.test_csv, windowing=windowing, normalize=True)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = TransformerTimeSeriesClassifier(
        input_dim=len(train_ds.feature_columns),
        num_classes=len(BEHAVIOR_CLASSES),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_ff,
        dropout=args.dropout,
        use_cls_token=args.use_cls_token,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = None if args.no_amp else torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_acc = -1.0
    best_ckpt_path = os.path.join(args.output_dir, "best_model.pt")
    epochs_without_improve = 0

    if not args.test_only:
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, scaler, criterion, grad_clip=args.grad_clip)
            val_loss, val_acc = evaluate(model, val_loader, device, criterion)

            print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

            # Early stopping on val_acc
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improve = 0
                torch.save({
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "val_acc": best_val_acc,
                }, best_ckpt_path)
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= args.early_stop_patience:
                    print("Early stopping triggered.")
                    break

    # Load best checkpoint if available
    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded best checkpoint with val_acc={ckpt.get('val_acc', 'NA')}")
    else:
        print("No checkpoint found; evaluating current model state.")

    # Evaluation on test set
    test_loss, test_acc = evaluate(model, test_loader, device, criterion)
    print(f"Test: loss={test_loss:.4f} acc={test_acc:.4f}")

    y_true, y_pred = predict_all(model, test_loader, device)
    print(classification_report(
        y_true,
        y_pred,
        labels=list(range(len(BEHAVIOR_CLASSES))),
        target_names=BEHAVIOR_CLASSES,
        zero_division=0,
    ))

    cm_out = os.path.join(args.output_dir, "confusion_matrix.png")
    save_confusion_matrix(y_true, y_pred, BEHAVIOR_CLASSES, cm_out, normalize=True)
    print(f"Saved confusion matrix to {cm_out}")


if __name__ == "__main__":
    main()


