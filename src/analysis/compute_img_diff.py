import numpy as np
from itertools import combinations
import json

# Define cohort data from the table (Same as in the paper)
cohorts = {
    "ADNI": {
        "manufacturer": {"Siemens": 461, "GE": 326, "Philips": 144},
        "field": {"1.5T": 409, "3.0T": 522},
        "labels": {"CN": 530, "AD": 401}
    },
    "OASIS-3": {
        "manufacturer": {"Siemens": 1095},
        "field": {"1.5T": 60, "3.0T": 1035},
        "labels": {"CN": 856, "AD": 239}
    },
    "OASIS-4": {
        "manufacturer": {"Siemens": 301},
        "field": {"1.5T": 20, "3.0T": 281},
        "labels": {"CN": 47, "AD": 254}
    },
    "AIBL": {
        "manufacturer": {"Siemens": 355},
        "field": {"1.5T": 83, "3.0T": 272},
        "labels": {"CN": 302, "AD": 53}
    },
    "BrainLAT": {
        "manufacturer": {"Siemens": 194, "GE": 8, "Philips": 127},
        "field": {"1.5T": 222, "3.0T": 141},
        "labels": {"CN": 158, "AD": 205}
    },
    "Nigerian Brain": {
        "manufacturer": {"GE": 18, "Hitachi": 27, "Toshiba": 5},
        "field": {"0.3T": 27, "1.5T": 23},
        "labels": {"CN": 27, "AD": 23}
    }
}

def get_distribution(counts_dict, all_keys):
    """Convert counts to a probability distribution over all_keys."""
    total = sum(counts_dict.values())
    return np.array([counts_dict.get(k, 0) / total for k in all_keys])

def jsd(p, q):
    """Compute Jensen-Shannon Divergence (base 2, so range 0-1)."""
    m = 0.5 * (p + q)
    # KL divergence with safe log (0*log(0) = 0)
    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

# Get all unique keys per dimension
all_manufacturers = sorted(set(k for c in cohorts.values() for k in c["manufacturer"]))
all_fields = sorted(set(k for c in cohorts.values() for k in c["field"]))
all_labels = sorted(set(k for c in cohorts.values() for k in c["labels"]))

cohort_names = list(cohorts.keys())
n = len(cohort_names)

# Compute pairwise JSD matrices
mfr_matrix = np.zeros((n, n))
field_matrix = np.zeros((n, n))
label_matrix = np.zeros((n, n))

for i, j in combinations(range(n), 2):
    ci, cj = cohort_names[i], cohort_names[j]

    p_mfr = get_distribution(cohorts[ci]["manufacturer"], all_manufacturers)
    q_mfr = get_distribution(cohorts[cj]["manufacturer"], all_manufacturers)
    mfr_matrix[i, j] = mfr_matrix[j, i] = jsd(p_mfr, q_mfr)

    p_field = get_distribution(cohorts[ci]["field"], all_fields)
    q_field = get_distribution(cohorts[cj]["field"], all_fields)
    field_matrix[i, j] = field_matrix[j, i] = jsd(p_field, q_field)

    p_label = get_distribution(cohorts[ci]["labels"], all_labels)
    q_label = get_distribution(cohorts[cj]["labels"], all_labels)
    label_matrix[i, j] = label_matrix[j, i] = jsd(p_label, q_label)

# Print results
def print_matrix(matrix, names, title):
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    # Header
    header = f"{'':>15}" + "".join(f"{name:>15}" for name in names)
    print(header)
    for i, name in enumerate(names):
        row = f"{name:>15}" + "".join(f"{matrix[i,j]:>15.4f}" for j in range(len(names)))
        print(row)

print_matrix(mfr_matrix, cohort_names, "JSD - Manufacturer")
print_matrix(field_matrix, cohort_names, "JSD - Magnetic Field Strength")
print_matrix(label_matrix, cohort_names, "JSD - Label Distribution")

# Also output as JSON for visualization
results = {
    "cohort_names": cohort_names,
    "manufacturer_jsd": mfr_matrix.tolist(),
    "field_jsd": field_matrix.tolist(),
    "label_jsd": label_matrix.tolist()
}
with open("jsd_results_prev.json", "w") as f:
    json.dump(results, f)

print("\nResults saved to jsd_results.json")