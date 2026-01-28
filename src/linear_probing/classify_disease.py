from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import pandas as pd
import random
import ot  # POT library for optimal transport

def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
set_seed()

class LinearClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.fc(x)


def train_each_test_all(df, n_splits=20, out_path="results_disease_classification.csv"):
    cohorts = df['cohort'].unique().tolist()
    FEATURE_COLS = [c for c in df.columns if c.startswith("Feature")]

    # Initialize results dictionary: train_cohort -> test_cohort -> list of scores
    results = []

    # Hyperparameter grid
    lr_list = [0.001, 0.005, 0.01]
    epoch_list = [25, 50, 100]
    batch_size_list = [64, 128]

    for train_cohort in cohorts:
        print(f"\n=== Training on {train_cohort} ===")

        train_df = df[df['cohort'] == train_cohort]
        X_np = train_df[FEATURE_COLS].values.astype(np.float32)
        y_np = train_df['label'].values.astype(np.float32)

        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=42)

        for split_idx, (train_idx, val_idx) in enumerate(sss.split(X_np, y_np)):
            X_train_np, X_val_np = X_np[train_idx], X_np[val_idx]
            y_train_np, y_val_np = y_np[train_idx], y_np[val_idx]

            X_train = torch.from_numpy(X_train_np).to(DEVICE)
            y_train = torch.from_numpy(y_train_np).to(DEVICE).unsqueeze(1)
            X_val = torch.from_numpy(X_val_np).to(DEVICE)
            y_val = torch.from_numpy(y_val_np).to(DEVICE).unsqueeze(1)

            # ---- Hyperparameter tuning on internal validation ----
            best_bal_acc = 0.0
            best_model_state = None
            best_lr, best_epochs, best_batch_size = None, None, None

            for lr in lr_list:
                for epochs in epoch_list:
                    for batch_size in batch_size_list:
                        model = LinearClassifier(input_dim=X_train.shape[1]).to(DEVICE)
                        criterion = nn.BCEWithLogitsLoss()
                        optimizer = optim.Adam(model.parameters(), lr=lr)
                        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

                        for _ in range(epochs):
                            model.train()
                            for batch_X, batch_y in train_loader:
                                optimizer.zero_grad()
                                loss = criterion(model(batch_X), batch_y)
                                loss.backward()
                                optimizer.step()

                        # Evaluate on validation split
                        model.eval()
                        with torch.no_grad():
                            logits = model(X_val)
                            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                            bal_acc = balanced_accuracy_score(y_val.cpu().numpy(), preds)

                        if bal_acc > best_bal_acc:
                            best_bal_acc = bal_acc
                            best_model_state = model.state_dict()
                            best_lr, best_epochs, best_batch_size = lr, epochs, batch_size

            print(f"Split {split_idx+1}: best lr={best_lr}, epochs={best_epochs}, batch_size={best_batch_size}, val balanced_acc={best_bal_acc:.3f}")

            # ---- Retrain best model on the full dataset of this cohort ----
            model = LinearClassifier(input_dim=X_train.shape[1]).to(DEVICE)
            model.load_state_dict(best_model_state)
            model.eval()
            
            # ---- Get training samples with label=0 for this split ----
            train_label_0_mask = (y_train_np == 0)
            X_train_label0_np = X_train_np[train_label_0_mask]

            # ---- Evaluate this model on all cohorts ----
            with torch.no_grad():
                for test_cohort in cohorts:
                    test_df = df[df['cohort'] == test_cohort]
                    X_test_np = test_df[FEATURE_COLS].values.astype(np.float32)
                    y_test_np = test_df['label'].values.astype(np.float32)

                    X_test = torch.from_numpy(X_test_np).to(DEVICE)
                    y_test = torch.from_numpy(y_test_np).to(DEVICE).unsqueeze(1)

                    logits = model(X_test)
                    preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                    bal_acc = balanced_accuracy_score(y_test.cpu().numpy(), preds)
                    f1 = f1_score(y_test.cpu().numpy(), preds)
                    auc = roc_auc_score(y_test.cpu().numpy(), torch.sigmoid(logits).cpu().numpy())

                    test_label_0_mask = (y_test_np == 0)
                    X_test_label0_np = X_test_np[test_label_0_mask]
                    
                    ## computing distance
                    if len(X_train_label0_np) > 0 and len(X_test_label0_np) > 0:
                        # Compute cost matrix (squared Euclidean distance)
                        M = ot.dist(X_train_label0_np, X_test_label0_np, metric='sqeuclidean')
                        
                        # Uniform distributions
                        a = np.ones(len(X_train_label0_np)) / len(X_train_label0_np)
                        b = np.ones(len(X_test_label0_np)) / len(X_test_label0_np)
                        
                        # Compute Wasserstein-2 distance
                        wasserstein_dist = ot.emd2(a, b, M)
                    else:
                        wasserstein_dist = np.nan

                    # Store results
                    results.append({
                        'train': train_cohort,
                        'test': test_cohort,
                        'split': split_idx,
                        'balanced_accuracy': bal_acc,
                        'f1_score': f1,
                        'auc': auc,
                        'wasserstein_dist': wasserstein_dist
                    })

    # ---- Convert nested results dict into a dataframe ----
    df = pd.DataFrame(data=results)

    df.to_csv(out_path, index=False)
    print(f"\nSaved cross-cohort matrix to {out_path}")

    return results, df

def main(args):
    df = pd.read_csv(args.features_csv)
    train_each_test_all(df, n_splits=args.num_splits, out_path=args.output_csv)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Disease Classification with Each-Train-All-Test")
    parser.add_argument('--features_csv', type=str, required=True, help='Path to CSV file containing features and labels')
    parser.add_argument('--output_csv', type=str, default='results_disease_classification.csv', help='Path to save the results CSV')
    parser.add_argument('--num_splits', type=int, default=20, help='Number of splits for StratifiedShuffleSplit')
    
    args = parser.parse_args()
    main(args)