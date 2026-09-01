"""
Domain-shift analysis (minimal version).

Quantifies the degree of domain shift of an XRD identification model between
two datasets (source domain MP / target domain COD or ICSD).

Pipeline:
    1. Load augmented spectra (*_aug.npy) from both datasets into a sample-level dataset.
    2. Extract deep features with a pretrained XRDNet (get_deep_feature).
    3. Evaluate classification accuracy on both domains (the consequence of domain shift).
    4. Reduce features to 2D with UMAP.
    5. Compute a per-class domain-shift score (bounded normalization based on
       per-dimension Wasserstein distances).
    6. Save the result CSVs and plot the global UMAP scatter.

Model:
    - Backbone: XRDNet in Model/basemodel.py (not redefined in this file).
    - Weights : Model/Checkpoints/basemodel_177.pth (contains best_params and model_state_dict).

Output files (saved to the current working directory):
    - domain_shift_scores_{cod|icsd}.csv           per-class shift scores (core result)
    - model_accuracy_{cod|icsd}.csv                overall / MP / target-domain accuracy
    - model_accuracy_by_class_{cod|icsd}.csv       per-class accuracy on the target domain
    - domain_shift_a_global_umap.png / .pdf        global UMAP scatter plot

Usage:
    python domain_shift_analysis.py
    Set the target domain, data directories, and mapping CSV in the CONFIG section below.

Note: this script is extracted from Domain_shift/transfer_see_plus.py, keeping only the
core domain-shift analysis, with the model definition and weights replaced by the external
basemodel.py and basemodel_177.pth.
"""
import os

os.environ['NUMBA_CACHE_DIR'] = ''  # Disable numba cache to avoid errors in multiprocessing/read-only environments.

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torch
import umap
from scipy.stats import wasserstein_distance

# ---------------------------------------------------------------------------
# Make Model/ importable by inserting its absolute path into sys.path.
# The path is resolved relative to this script, so it works from any CWD.
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.join(_BASE_DIR, '..', 'Model')
if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)

from basemodel import XRDNet

# ==================== Configuration ====================
N_CLASSES = 177                         # Number of output classes (basemodel_177.pth is a 177-class model).
MODEL_WEIGHTS = os.path.join(_MODEL_DIR, 'Checkpoints', 'basemodel_177.pth')

TARGET_DOMAIN = "ICSD"                  # Target domain: "COD" or "ICSD".

# Directories holding the augmented spectra (*_aug.npy). These data files are not
# shipped with the code; adjust them to your actual paths before running.
if TARGET_DOMAIN.upper() == "COD":
    MP_DIR = "../mp_cod_icsd_overlap/cod_mp_overlap/from_mp"
    TARGET_DIR = "../mp_cod_icsd_overlap/cod_mp_overlap/from_cod"
    TARGET_LABEL = "COD"
    TARGET_LOWER = "cod"
elif TARGET_DOMAIN.upper() == "ICSD":
    MP_DIR = "../mp_cod_icsd_overlap/icsd_mp_overlap/from_mp"
    TARGET_DIR = "../mp_cod_icsd_overlap/icsd_mp_overlap/from_icsd"
    TARGET_LABEL = "ICSD"
    TARGET_LOWER = "icsd"
else:
    raise ValueError(f"Unsupported target domain: {TARGET_DOMAIN}. Use COD or ICSD.")

# Compound -> class-label mapping (from the clustering output compound_cluster_summary.csv).
LABEL_CSV = "../Clustering/compound_cluster_summary.csv"
COL_COMPOUND = "Compound"
COL_CLASS = "cluster_id"

np.random.seed(42)

# ==================== 1. Load the label CSV and build the mapping ====================
print("Reading the label CSV ...")
df_labels = pd.read_csv(LABEL_CSV)
print(f"CSV columns: {df_labels.columns.tolist()}")
compound_to_class = dict(zip(df_labels[COL_COMPOUND], df_labels[COL_CLASS]))


# ==================== 2. Load .npy files and build the sample-level dataset ====================
def load_dataset_from_dir(dir_path, source_name, apply_subsample=False):
    """
    Return sample-level data:
        X: shape = (N_samples, 4501, 1)
        y: shape = (N_samples,)
        compound_names: shape = (N_samples,)
    """
    data_list = []
    label_list = []
    compound_list = []
    missing_count = 0
    print(f"\nLoading data from {source_name} ...")

    for f in os.listdir(dir_path):
        if not f.endswith("_aug.npy"):
            continue
        compound_name = f.replace("_aug.npy", "")
        if compound_name not in compound_to_class:
            missing_count += 1
            continue
        label = compound_to_class[compound_name]
        data = np.load(os.path.join(dir_path, f))

        # Ensure the shape is (n_aug, 4501, 1).
        if data.ndim == 2:
            data = data[..., None]

        # MP subsampling: keep 20 of every 50 spectra to cap the sample size.
        if apply_subsample:
            window_size = 50
            sample_num = 20
            total_windows = data.shape[0] // window_size
            selected_indices = []
            for i in range(total_windows):
                start_idx = i * window_size
                end_idx = start_idx + window_size
                rand_idx = np.random.choice(
                    np.arange(start_idx, end_idx),
                    size=sample_num,
                    replace=False,
                )
                selected_indices.extend(rand_idx)
            data = data[selected_indices]

        data_list.append(data.astype(np.float32))
        label_list.extend([label] * data.shape[0])
        compound_list.extend([compound_name] * data.shape[0])

    if missing_count > 0:
        print(f"  [Warning] {missing_count} files in {source_name} have no matching label in the CSV and were skipped.")
    if len(data_list) == 0:
        raise ValueError(f"{source_name} loaded no data; check the path and the CSV mapping.")

    X = np.concatenate(data_list, axis=0)
    y = np.array(label_list)
    compound_names = np.array(compound_list)
    print(f"{source_name}: {X.shape[0]} samples loaded")
    print(f"{source_name} X shape: {X.shape}")
    print(f"{source_name} classes: {len(np.unique(y))}")
    return X, y, compound_names


X_mp, y_mp_raw, compound_mp = load_dataset_from_dir(MP_DIR, "from_mp", apply_subsample=True)
X_target, y_target_raw, compound_target = load_dataset_from_dir(
    TARGET_DIR, f"from_{TARGET_LOWER}", apply_subsample=False
)

print("\nMP data:", X_mp.shape)
print(f"{TARGET_LABEL} data:", X_target.shape)

# ==================== 3. Merge the source and target data ====================
X_all = np.concatenate([X_mp, X_target], axis=0)
class_labels = np.concatenate([y_mp_raw, y_target_raw], axis=0)   # Raw class labels.
source_labels = np.array([0] * len(X_mp) + [1] * len(X_target))   # 0 = MP (source), 1 = target domain.

class_labels = class_labels.astype(np.int64)
if class_labels.min() < 0 or class_labels.max() >= N_CLASSES:
    raise ValueError(
        f"Class labels must be in 0~{N_CLASSES - 1}, got "
        f"{class_labels.min()}~{class_labels.max()}"
    )

unique_classes = pd.unique(class_labels)
y_onehot = np.eye(N_CLASSES, dtype=np.float32)[class_labels]

print("\nLoading the model config and weights ...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(MODEL_WEIGHTS, map_location=device, weights_only=False)


class XRDInferenceDataset(Dataset):
    def __init__(self, X, onehot_labels, raw_labels, source_labels):
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(f"X must be a 3-D array, got shape={X.shape}")
        if X.shape[-1] == 1:
            X = np.transpose(X, (0, 2, 1))
        self.X = torch.from_numpy(X).float()
        self.onehot_labels = torch.from_numpy(onehot_labels).float()
        self.raw_labels = np.asarray(raw_labels)
        self.source_labels = np.asarray(source_labels)

    def __len__(self):
        return len(self.raw_labels)

    def __getitem__(self, index):
        return (
            self.X[index],
            self.onehot_labels[index],
            int(self.raw_labels[index]),
            int(self.source_labels[index]),
        )


# ==================== 4. Build the DataLoader ====================
print("\nBuilding the Dataset and DataLoader ...")
dataset = XRDInferenceDataset(X_all, y_onehot, class_labels, source_labels)
dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)


# ==================== 5. Load the model and extract deep features ====================
print("\nLoading the model and extracting features ...")
if "best_params" not in checkpoint:
    raise KeyError("checkpoint has no 'best_params'; cannot build XRDNet.")
if "model_state_dict" not in checkpoint:
    raise KeyError("checkpoint has no 'model_state_dict'; cannot load model weights.")

model = XRDNet(n_phases=N_CLASSES, best_params=checkpoint["best_params"])
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()


# ==================== 6. Classification accuracy (consequence of domain shift) ====================
def evaluate_accuracy(dataloader, model, device):
    total_correct = 0
    total_count = 0
    mp_correct = 0
    mp_count = 0
    target_correct = 0
    target_count = 0

    target_per_class_correct = np.zeros(N_CLASSES, dtype=np.int64)
    target_per_class_count = np.zeros(N_CLASSES, dtype=np.int64)

    with torch.no_grad():
        for batch_data in dataloader:
            if not isinstance(batch_data, (tuple, list)) or len(batch_data) < 4:
                raise ValueError("Dataset must return (X, onehot labels, raw class labels, source labels).")
            batch_x = batch_data[0]
            batch_raw_labels = np.asarray(batch_data[2], dtype=np.int64)
            batch_source = np.asarray(batch_data[3], dtype=np.int64)
            batch_x = batch_x.to(device).float()
            outputs = model(batch_x)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            targets = batch_raw_labels
            batch_size = len(targets)
            correct = preds == targets

            total_correct += int(correct.sum())
            total_count += batch_size

            mp_mask_batch = batch_source == 0
            target_mask_batch = batch_source == 1

            if mp_mask_batch.any():
                mp_correct += int(correct[mp_mask_batch].sum())
                mp_count += int(mp_mask_batch.sum())
            if target_mask_batch.any():
                target_correct += int(correct[target_mask_batch].sum())
                target_count += int(target_mask_batch.sum())

                target_targets = targets[target_mask_batch]
                target_correct_mask = correct[target_mask_batch]
                np.add.at(target_per_class_count, target_targets, 1)
                np.add.at(target_per_class_correct, target_targets, target_correct_mask.astype(np.int64))

    return {
        "accuracy_all": total_correct / max(total_count, 1),
        "accuracy_mp": mp_correct / max(mp_count, 1),
        "accuracy_target": target_correct / max(target_count, 1),
        "count_all": total_count,
        "count_mp": mp_count,
        "count_target": target_count,
        "target_per_class_correct": target_per_class_correct,
        "target_per_class_count": target_per_class_count,
    }


accuracy_stats = evaluate_accuracy(dataloader, model, device)
accuracy_summary = pd.DataFrame([
    {
        "target_domain": TARGET_LABEL,
        "accuracy_all": accuracy_stats["accuracy_all"],
        "accuracy_mp": accuracy_stats["accuracy_mp"],
        "accuracy_target": accuracy_stats["accuracy_target"],
        "count_all": accuracy_stats["count_all"],
        "count_mp": accuracy_stats["count_mp"],
        "count_target": accuracy_stats["count_target"],
    }
])
accuracy_csv = f"model_accuracy_{TARGET_LOWER}.csv"
accuracy_summary.to_csv(accuracy_csv, index=False, encoding="utf-8-sig")

per_class_df = pd.DataFrame({
    "class_label": np.arange(N_CLASSES, dtype=np.int64),
    "correct": accuracy_stats["target_per_class_correct"],
    "count": accuracy_stats["target_per_class_count"],
})
per_class_df = per_class_df[per_class_df["count"] > 0].copy()
per_class_df["accuracy"] = per_class_df["correct"] / per_class_df["count"]
per_class_df = per_class_df.sort_values(
    ["accuracy", "count", "class_label"], ascending=[True, True, True]
).reset_index(drop=True)
per_class_csv = f"model_accuracy_by_class_{TARGET_LOWER}.csv"
per_class_df.to_csv(per_class_csv, index=False, encoding="utf-8-sig")

print("\nClassification accuracy:")
print(f"  Overall accuracy: {accuracy_stats['accuracy_all']:.4f}  (n={accuracy_stats['count_all']})")
print(f"  MP accuracy:      {accuracy_stats['accuracy_mp']:.4f}  (n={accuracy_stats['count_mp']})")
print(f"  {TARGET_LABEL} accuracy: {accuracy_stats['accuracy_target']:.4f}  (n={accuracy_stats['count_target']})")
print(f"  Accuracy results saved to: {os.path.abspath(accuracy_csv)}")
print(f"  Per-class accuracy saved to: {os.path.abspath(per_class_csv)}")
print("\nTop 10 classes with the lowest accuracy:")
print(per_class_df[["class_label", "accuracy", "count", "correct"]].head(10))

# Extract deep features (the activation right after the second-to-last ReLU of the classifier).
features_list = []
with torch.no_grad():
    for batch_data in dataloader:
        if not isinstance(batch_data, (tuple, list)):
            raise ValueError("Dataset must return at least batch_x.")
        batch_x = batch_data[0]
        batch_x = batch_x.to(device).float()
        feat = model.get_deep_feature(batch_x)
        features_list.append(feat.cpu())

X_features = torch.cat(features_list, dim=0).numpy()
print(f"Feature extraction done, feature dimension: {X_features.shape}")


# ==================== 7. UMAP 2D reduction ====================
print("\nRunning UMAP 2D reduction ...")
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=30,
    min_dist=0.1,
    metric="cosine",
    random_state=42,
)
X_2d = reducer.fit_transform(X_features)
print("UMAP done:", X_2d.shape)


# ==================== 8. Compute per-class domain-shift scores ====================
def standardize_features(X, eps=1e-8):
    """Standardize features for domain-shift scoring (do not compute quantitative scores on UMAP coordinates)."""
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    return (X - mean) / (std + eps)


X_metric = standardize_features(X_features)
records = []
eps = 1e-8

for cls in unique_classes:
    idx_mp = (class_labels == cls) & (source_labels == 0)
    idx_target = (class_labels == cls) & (source_labels == 1)

    if idx_mp.sum() == 0 or idx_target.sum() == 0:
        continue

    F_mp = X_metric[idx_mp]
    F_target = X_metric[idx_target]

    mu_mp_f = F_mp.mean(axis=0)
    mu_target_f = F_target.mean(axis=0)

    # Mean distance to the centroid = "radius/compactness", used for normalization.
    r_mp = np.linalg.norm(F_mp - mu_mp_f, axis=1).mean()
    r_target = np.linalg.norm(F_target - mu_target_f, axis=1).mean()

    # L2 norm of per-dimension 1D Wasserstein distances, as an approximation of the
    # high-dimensional distributional distance.
    wd_per_dim = np.array([
        wasserstein_distance(F_mp[:, d], F_target[:, d]) for d in range(F_mp.shape[1])
    ])
    feature_shift = np.linalg.norm(wd_per_dim)

    pooled_radius = 0.5 * (r_mp + r_target)
    shift_score = feature_shift / (feature_shift + pooled_radius + eps)

    # Project the centroids into UMAP 2D space (saved in the result CSV).
    centroids_high = np.vstack([mu_mp_f, mu_target_f])
    centroids_2d = reducer.transform(centroids_high)
    mu_mp_u = centroids_2d[0]
    mu_target_u = centroids_2d[1]

    # Wasserstein distance between the two domains in UMAP space.
    U_mp_2d = X_2d[idx_mp]
    U_target_2d = X_2d[idx_target]
    wd_u1 = wasserstein_distance(U_mp_2d[:, 0], U_target_2d[:, 0])
    wd_u2 = wasserstein_distance(U_mp_2d[:, 1], U_target_2d[:, 1])
    umap_shift = np.linalg.norm([wd_u1, wd_u2])

    records.append({
        "class_label": cls,
        "n_mp": int(idx_mp.sum()),
        "n_target": int(idx_target.sum()),
        "mp_umap1": mu_mp_u[0],
        "mp_umap2": mu_mp_u[1],
        "target_umap1": mu_target_u[0],
        "target_umap2": mu_target_u[1],
        "feature_shift": feature_shift,
        "r_mp": r_mp,
        "r_target": r_target,
        "shift_score": shift_score,
        "umap_shift": umap_shift,
    })

shift_df = pd.DataFrame(records)
shift_df = shift_df.sort_values("shift_score", ascending=False).reset_index(drop=True)
shift_df.to_csv(f"domain_shift_scores_{TARGET_LOWER}.csv", index=False, encoding="utf-8-sig")

print("\nDomain-shift score computation done.")
print(f"Valid classes: {len(shift_df)}")
print(f"Results saved to: {os.path.abspath(f'domain_shift_scores_{TARGET_LOWER}.csv')}")
print("\nTop 10 classes by shift_score:")
print(shift_df[["class_label", "n_mp", "n_target", "shift_score", "feature_shift", "umap_shift"]].head(10))


# ==================== 9. Plot the global UMAP scatter ====================
print("\nPlotting the global UMAP scatter ...")

C_TEXT = "#334155"

mp_mask = source_labels == 0
target_mask = source_labels == 1

num_classes = len(unique_classes)
# Generate enough colors on the HSV wheel and shuffle them so nearby classes differ.
cmap_colors = plt.cm.hsv(np.linspace(0, 1, num_classes))
np.random.shuffle(cmap_colors)
class_to_color = {cls: cmap_colors[i] for i, cls in enumerate(unique_classes)}

colors_mp = [class_to_color[c] for c in class_labels[mp_mask]]
colors_target = [class_to_color[c] for c in class_labels[target_mask]]

fig_a, ax_a = plt.subplots(figsize=(5, 4))
ax_a.scatter(X_2d[mp_mask, 0], X_2d[mp_mask, 1], c=colors_mp, s=5, marker="o",
             alpha=0.40, edgecolors="none", rasterized=True, label="MP (Circle)")
ax_a.scatter(X_2d[target_mask, 0], X_2d[target_mask, 1], c=colors_target, s=7, marker="^",
             alpha=0.60, edgecolors="black", linewidths=0.3, rasterized=True,
             label=f"{TARGET_LABEL} (Triangle)")
ax_a.set_xlabel("UMAP 1", fontsize=10, color=C_TEXT)
ax_a.set_ylabel("UMAP 2", fontsize=10, color=C_TEXT)
ax_a.tick_params(axis="both", labelsize=7, colors=C_TEXT)
for spine in ax_a.spines.values():
    spine.set_edgecolor(C_TEXT)
    spine.set_linewidth(0.8)
ax_a.legend(frameon=False, fontsize=8, loc="best", title="Database", title_fontsize=9)
plt.tight_layout()
fig_a.savefig("domain_shift_a_global_umap.png", dpi=600, bbox_inches="tight")
fig_a.savefig("domain_shift_a_global_umap.pdf", bbox_inches="tight")
plt.close(fig_a)

print("\nGlobal UMAP scatter saved (PNG & PDF).")
