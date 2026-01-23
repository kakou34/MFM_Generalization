import itertools
import argparse
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simple linear classifier
class LinearClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.fc(x)

# Function to run one pairwise experiment between any two cohorts
def run_pairwise_experiment(df, cohort_a, cohort_b, n_splits=20, epochs=100, lr=1e-3):
    results_bacc = []
    results_f1 = []
    results_rocauc = []

    # Filter label==0 for each cohort
    a_samples = df[(df['cohort'] == cohort_a) & (df['label'] == 0)]
    b_samples = df[(df['cohort'] == cohort_b) & (df['label'] == 0)]


    FEATURE_COLS = [c for c in df.columns if c.startswith("Feature")]
    # Combine data
    data = pd.concat([a_samples, b_samples])
    X = data[FEATURE_COLS].values.astype(np.float32)
    y = (data['cohort'] == cohort_a).astype(np.float32).values  # cohort_a = 1, cohort_b = 0

    X_tensor = torch.from_numpy(X).to(DEVICE)
    y_tensor = torch.from_numpy(y).to(DEVICE).unsqueeze(1)


    for split_idx in range(n_splits):
        train_idx, test_idx = train_test_split(np.arange(len(X)), test_size=0.5, stratify=y, random_state=split_idx)
        X_train, X_test = X_tensor[train_idx], X_tensor[test_idx]
        y_train, y_test = y_tensor[train_idx], y_tensor[test_idx]

        # Special case: oversample NigerianBrain if used
        if cohort_a == 'NigerianBrain' or cohort_b == 'NigerianBrain':
            minority_label = 0 if cohort_a == 'NigerianBrain' else 1
            minority_train = X_train[y_train.squeeze() == minority_label]
            majority_train = X_train[y_train.squeeze() != minority_label]
            # Oversample minority to match majority size
            if len(minority_train) > 0:
                n_majority = len(majority_train)
                # Randomly pick indices from minority_train with replacement
                indices = torch.randint(low=0, high=len(minority_train), size=(n_majority,), device=DEVICE)
                minority_oversampled = minority_train[indices]

                # Concatenate oversampled minority with majority
                X_train = torch.cat([majority_train, minority_oversampled], dim=0)
                y_train = torch.cat([
                    torch.full((n_majority, 1), 1 - minority_label, device=DEVICE, dtype=torch.float32),
                    torch.full((n_majority, 1), minority_label, device=DEVICE, dtype=torch.float32)
                ], dim=0)

        # Shuffle training set
        perm = torch.randperm(len(X_train))
        X_train, y_train = X_train[perm], y_train[perm]

        # Initialize and train model
        model = LinearClassifier(input_dim=X.shape[1]).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            logits = model(X_test)
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
            bal_acc = balanced_accuracy_score(y_test.cpu().numpy(), preds)
            f1 = f1_score(y_test.cpu().numpy(), preds)
            roc_auc = roc_auc_score(y_test.cpu().numpy(), torch.sigmoid(logits).cpu().numpy())
            
            results_bacc.append(bal_acc)
            results_f1.append(f1)
            results_rocauc.append(roc_auc)

    return results_bacc, results_f1, results_rocauc


# ---- Main execution ----

def main(args):
    df = pd.read_csv(args.features_csv)

    cohorts = pd.unique(df['cohort'])
    all_results = {}

    print(itertools.combinations(cohorts, 2))

    # Generate all pairwise combinations (unordered, no repetition)
    for cohort_a, cohort_b in itertools.combinations(cohorts, 2):
        key = f"{cohort_a}_vs_{cohort_b}"
        print(f"\nRunning experiment: {cohort_a} vs {cohort_b}")
        bal_accs, f1_scores, roc_aucs = run_pairwise_experiment(df, cohort_a, cohort_b, n_splits=args.num_splits, epochs=args.num_epochs, lr=args.learning_rate)
        all_results[key] = {
            "balanced_accuracy": bal_accs,
            "f1_score": f1_scores,
            "roc_auc": roc_aucs,
        }

    # Save results
    rows = []
    for key, metrics in all_results.items():
        for split_idx, (bal, f1, auc) in enumerate(
            zip(
                metrics["balanced_accuracy"],
                metrics["f1_score"],
                metrics["roc_auc"],
            )
        ):
            rows.append({
                "experiment": key,
                "split": split_idx,
                "balanced_accuracy": bal,
                "f1_score": f1,
                "roc_auc": auc,
            })

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Perform pairwise cohort classification using linear probing"
    )

    parser.add_argument(
        "--features_csv",
        type=str,
        required=True,
        help="Path to the input CSV file"
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to the output CSV file"
    )
    
    parser.add_argument(
        "--num_epochs",
        type=int,
        required=True,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        required=True,
        help="Learning rate for optimizer"
    )
    
    parser.add_argument(
        "--num_splits",
        type=int,
        required=True,
        help="Number of train-test splits"
    )
    
    args = parser.parse_args()
    
    main(args)

    


