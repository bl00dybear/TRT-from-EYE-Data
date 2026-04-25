import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from torch.nn import HuberLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from config_model import ModelConfig


class TabularDataset(Dataset):
    def __init__(self, num_x, y=None):
        self.num_x = torch.tensor(num_x, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.num_x.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.num_x[idx]
        return self.num_x[idx], self.y[idx]


class TabularRegressor(nn.Module):
    def __init__(self, num_numeric, dropout):
        super().__init__()

        self.fc1 = nn.Linear(num_numeric, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.out = nn.Linear(128, 1)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, num_x):
        x = self.drop(self.relu(self.bn1(self.fc1(num_x))))
        x = self.drop(self.relu(self.bn2(self.fc2(x))))
        x = self.drop(self.relu(self.bn3(self.fc3(x))))
        return self.out(x).squeeze(1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def standardize_numeric(train_df, val_df, test_df, num_cols):
    means = train_df[num_cols].mean()
    stds = train_df[num_cols].std().replace(0, 1.0)

    train_scaled = ((train_df[num_cols] - means) / stds).astype(np.float32)
    val_scaled = ((val_df[num_cols] - means) / stds).astype(np.float32)
    test_scaled = ((test_df[num_cols] - means) / stds).astype(np.float32)

    train_df[num_cols] = train_scaled
    val_df[num_cols] = val_scaled
    test_df[num_cols] = test_scaled


def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = max(0.0, r2_score(y_true, y_pred))

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        pearson = 0.0
    else:
        p = pearsonr(y_true, y_pred)[0]
        pearson = float(abs(p)) if not np.isnan(p) else 0.0

    score = 100.0 * (r2 + pearson) / 2.0
    return rmse, r2, pearson, score


def evaluate(model, loader, device):
    model.eval()
    preds = []
    truths = []

    with torch.no_grad():
        for num_x, y in loader:
            num_x = num_x.to(device)
            y = y.to(device)

            out = model(num_x)
            out = torch.expm1(out).clamp(min=0)
            y_real = torch.expm1(y).clamp(min=0)

            preds.extend(out.cpu().numpy())
            truths.extend(y_real.cpu().numpy())

    preds = np.array(preds)
    truths = np.array(truths)
    return compute_metrics(truths, preds)


def train(model, train_loader, valid_loader, optimizer, criterion, scheduler, name, cfg):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)

    best_valid_score = -1.0
    early_stop_cnt = 0
    best_metrics = None

    for epoch in range(cfg.epochs):
        model.train()

        print(f"Epoch {epoch + 1}/{cfg.epochs}:")

        total_loss = 0.0
        total_samples = 0

        for num_x, y in train_loader:
            num_x = num_x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            output = model(num_x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)

        train_loss = total_loss / max(1, total_samples)
        print(f"        Train loss: {train_loss}")

        valid_rmse, valid_r2, valid_pearson, valid_score = evaluate(model, valid_loader, device)
        print(f"        Valid RMSE: {valid_rmse}, Valid R2: {valid_r2}, Valid Pearson: {valid_pearson}, Valid Score: {valid_score}")

        for param_group in optimizer.param_groups:
            print(f"        Learning rate: {param_group['lr']}")

        scheduler.step(valid_score)

        if valid_score > best_valid_score:
            best_valid_score = valid_score
            early_stop_cnt = 0
            best_metrics = (valid_rmse, valid_r2, valid_pearson, valid_score)
            torch.save(model.state_dict(), os.path.join(cfg.output_dir, name))
            print(f"             New best: {valid_score}")
        else:
            early_stop_cnt += 1

        if early_stop_cnt >= cfg.estop_limit:
            print(f"\n        Early stopped {best_valid_score}")
            break

    return best_metrics


def test(model, test_loader, device):
    model.eval()
    predicts = []

    with torch.no_grad():
        for num_x in test_loader:
            num_x = num_x.to(device)

            outputs = model(num_x)
            outputs = torch.expm1(outputs).clamp(min=0)
            predicts.extend(outputs.cpu().numpy())

    return np.array(predicts)


def main():
    cfg = ModelConfig()
    set_seed(cfg.random_state)

    os.makedirs(cfg.output_dir, exist_ok=True)

    train_df = pd.read_csv(cfg.train_path)
    test_df = pd.read_csv(cfg.test_path)

    unique_ids = train_df[cfg.participant_col].astype(str).unique().to_numpy(copy=True)
    np.random.seed(cfg.random_state)
    np.random.shuffle(unique_ids)

    n_val = max(1, int(len(unique_ids) * cfg.val_fraction_participants))
    val_participants = set(unique_ids[-n_val:])

    val_mask = train_df[cfg.participant_col].astype(str).isin(val_participants)

    train_part = train_df.loc[~val_mask].copy()
    valid_part = train_df.loc[val_mask].copy()

    y_train = np.log1p(train_part[cfg.target_col].values.astype(np.float32))
    y_valid = np.log1p(valid_part[cfg.target_col].values.astype(np.float32))

    numeric_cols = [
        col
        for col in cfg.numeric_features
        if col in train_part.columns and col in valid_part.columns and col in test_df.columns
    ]

    for col in numeric_cols:
        train_part[col] = pd.to_numeric(train_part[col], errors="coerce")
        valid_part[col] = pd.to_numeric(valid_part[col], errors="coerce")
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce")

    train_part[numeric_cols] = train_part[numeric_cols].fillna(0)
    valid_part[numeric_cols] = valid_part[numeric_cols].fillna(0)
    test_df[numeric_cols] = test_df[numeric_cols].fillna(0)

    standardize_numeric(train_part, valid_part, test_df, numeric_cols)

    x_train_num = train_part[numeric_cols].values
    x_valid_num = valid_part[numeric_cols].values
    x_test_num = test_df[numeric_cols].values

    train_dataset = TabularDataset(x_train_num, y_train)
    valid_dataset = TabularDataset(x_valid_num, y_valid)
    test_dataset = TabularDataset(x_test_num)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model = TabularRegressor(
        num_numeric=len(numeric_cols),
        dropout=cfg.dropout,
    )

    criterion = HuberLoss(delta=1.0)
    optimizer = Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=cfg.scheduler_patience, factor=cfg.scheduler_factor)

    best_metrics = train(model, train_loader, valid_loader, optimizer, criterion, scheduler, cfg.model_filename, cfg)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.load_state_dict(torch.load(os.path.join(cfg.output_dir, cfg.model_filename), map_location=device, weights_only=True))
    model.to(device)

    test_preds = test(model, test_loader, device)

    if cfg.submission_id_col in test_df.columns:
        datapoint_ids = test_df[cfg.submission_id_col].values
    else:
        datapoint_ids = np.arange(len(test_df))

    submission_df = pd.DataFrame(
        {
            "subtaskID": np.ones(len(test_df), dtype=int),
            "datapointID": datapoint_ids,
            "answer": test_preds,
        }
    )

    submission_path = os.path.join(cfg.output_dir, cfg.submission_filename)

    submission_df["datapointID"] = pd.to_numeric(submission_df["datapointID"], errors="coerce")
    submission_df = submission_df.sort_values("datapointID").reset_index(drop=True)

    submission_df.to_csv(submission_path, index=False)

    report_path = os.path.join(cfg.output_dir, cfg.validation_report_filename)
    with open(report_path, "w") as f:
        if best_metrics is not None:
            rmse, r2, pearson, score = best_metrics
            f.write(f"RMSE: {rmse:.6f}\n")
            f.write(f"R2: {r2:.6f}\n")
            f.write(f"Pearson: {pearson:.6f}\n")
            f.write(f"Hackathon Score: {score:.6f}\n")
        f.write(f"Rows submission: {len(submission_df)}\n")
        f.write(f"Prediction mean: {submission_df['answer'].mean():.6f}\n")
        f.write(f"Prediction std: {submission_df['answer'].std():.6f}\n")

    print("\n=== VALIDATION METRICS ===")
    if best_metrics is not None:
        rmse, r2, pearson, score = best_metrics
        print(f"RMSE: {rmse:.4f}")
        print(f"R2: {r2:.4f}")
        print(f"Pearson: {pearson:.4f}")
        print(f"Hackathon Score: {score:.4f}")

    print("\n=== SUBMISSION SUMMARY ===")
    print(f"Rows: {len(submission_df)}")
    print(
        f"Predicted TRT - mean: {test_preds.mean():.2f}, std: {test_preds.std():.2f}, min: {test_preds.min():.2f}, max: {test_preds.max():.2f}"
    )
    print(f"Saved to: {submission_path}")


if __name__ == "__main__":
    main()
