"""
Attention-based Multiple Instance Learning (MIL) for semi-supervised regional prediction.

Bag = one week (all cells). Bag label = week-level label (from config).
Instances = cells; instance labels unknown. Model learns which cells contribute to bag label.
Attention weights provide soft per-cell importance (discovered labels).
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class AttentionMIL(nn.Module):
    """
    Attention-based MIL. Bag = week (cells). Predicts bag label from instances.
    """

    def __init__(self, n_features: int, n_classes: int, embed_dim: int = 64, attn_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )
        self.attention_V = nn.Linear(embed_dim, attn_dim)
        self.attention_W = nn.Linear(attn_dim, 1)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (n_instances, n_features)
        Returns: (logits, attention_weights)
        """
        h = self.encoder(x)  # (n_instances, embed_dim)
        a = self.attention_W(torch.tanh(self.attention_V(h)))  # (n_instances, 1)
        a = torch.softmax(a, dim=0)  # attention over instances
        z = (a * h).sum(dim=0)  # (embed_dim,) bag embedding
        logits = self.classifier(z)  # (n_classes,)
        return logits, a.squeeze(-1)


class MILDataset(Dataset):
    """Dataset of bags (weeks). Each bag = (n_cells, n_features)."""

    def __init__(self, bags: list[np.ndarray], labels: np.ndarray):
        self.bags = bags
        self.labels = labels

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        return torch.from_numpy(self.bags[idx].astype(np.float32)), self.labels[idx]


def collate_bags(batch):
    """Return list of (x, y) - variable bag sizes, no batching across bags."""
    return batch


def bags_from_regional(
    X_local: np.ndarray,
    labels: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Convert (n_cells, n_weeks, n_features) to list of bags.
    Each bag = (n_cells, n_features) for one week.
    """
    n_cells, n_weeks, n_feat = X_local.shape
    bags = []
    for t in range(n_weeks):
        bag = X_local[:, t, :]  # (n_cells, n_features)
        # Replace NaN with 0
        bag = np.nan_to_num(bag, nan=0.0, posinf=0.0, neginf=0.0)
        bags.append(bag)
    return bags, labels.astype(np.int64)


def train_and_evaluate_mil(
    bags_train: list[np.ndarray],
    y_train: np.ndarray,
    bags_test: list[np.ndarray] | None = None,
    y_test: np.ndarray | None = None,
    class_names: list[str] | None = None,
    n_epochs: int = 100,
    lr: float = 1e-3,
    device: str | None = None,
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Train Attention MIL on bags. Returns metrics and optionally attention weights.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for MIL. Install with: pip install torch")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    n_features = bags_train[0].shape[1]
    n_classes = len(np.unique(y_train))
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    # Scale features
    scaler = StandardScaler()
    all_train = np.vstack(bags_train)
    scaler.fit(all_train)
    bags_train_scaled = [scaler.transform(b) for b in bags_train]

    model = AttentionMIL(n_features, n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    torch.manual_seed(random_state)
    dataset = MILDataset(bags_train_scaled, y_train)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_bags)

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        for batch in loader:
            for x, y in batch:
                x = x.to(device)
                y = torch.tensor([y], dtype=torch.long, device=device)
                optimizer.zero_grad()
                logits, _ = model(x)
                loss = criterion(logits.unsqueeze(0), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs} loss={total_loss/len(dataset):.4f}")

    # Evaluate
    model.eval()
    y_train_pred = []
    with torch.no_grad():
        for bag in bags_train_scaled:
            x = torch.from_numpy(bag.astype(np.float32)).to(device)
            logits, _ = model(x)
            y_train_pred.append(logits.argmax().item())

    train_accuracy = accuracy_score(y_train, y_train_pred)
    results = {
        "train_accuracy": train_accuracy,
        "model": model,
        "scaler": scaler,
    }

    if bags_test is not None and y_test is not None:
        bags_test_scaled = [scaler.transform(b) for b in bags_test]
        y_test_pred = []
        with torch.no_grad():
            for bag in bags_test_scaled:
                x = torch.from_numpy(bag.astype(np.float32)).to(device)
                logits, _ = model(x)
                y_test_pred.append(logits.argmax().item())

        results["test_accuracy"] = accuracy_score(y_test, y_test_pred)
        n_classes = len(class_names)
        results["classification_report_test"] = classification_report(
            y_test, y_test_pred, target_names=class_names, labels=range(n_classes), zero_division=0
        )
        results["confusion_matrix_test"] = confusion_matrix(
            y_test, y_test_pred, labels=range(n_classes)
        )

    n_classes = len(class_names)
    results["classification_report_train"] = classification_report(
        y_train, y_train_pred, target_names=class_names, labels=range(n_classes), zero_division=0
    )

    return results


def train_and_evaluate_mil_cv(
    bags: list[np.ndarray],
    y: np.ndarray,
    class_names: list[str] | None = None,
    n_splits: int = 5,
    n_epochs: int = 80,
    lr: float = 1e-3,
    device: str | None = None,
    random_state: int = 42,
    verbose: bool = False,
) -> dict:
    """Time-series CV for MIL: train on past weeks, test on future."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for MIL. Install with: pip install torch")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    accs = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(bags)):
        bags_train = [bags[i] for i in train_idx]
        y_train = y[train_idx]
        bags_test = [bags[i] for i in test_idx]
        y_test = y[test_idx]

        fold_results = train_and_evaluate_mil(
            bags_train, y_train,
            bags_test, y_test,
            class_names=class_names,
            n_epochs=n_epochs,
            lr=lr,
            device=device,
            random_state=random_state + fold,
            verbose=verbose,
        )
        accs.append(fold_results["test_accuracy"])

    # Final fit on all data for reporting
    final = train_and_evaluate_mil(
        bags, y,
        class_names=class_names,
        n_epochs=n_epochs,
        lr=lr,
        device=device,
        random_state=random_state,
        verbose=verbose,
    )

    return {
        "cv_accuracy_mean": np.mean(accs),
        "cv_accuracy_std": np.std(accs),
        "train_accuracy": final["train_accuracy"],
        "classification_report": final["classification_report_train"],
        "model": final["model"],
        "scaler": final["scaler"],
    }
