"""Inference script: classify RRUFF XRD spectra with a trained model.

This script loads a trained XRDNet checkpoint together with its label map,
runs inference over the .npy spectra placed in ``RRUFF_processed``, and writes
the top-k predicted compounds to JSON and CSV files.

Supported models (``DATASET_NAME``):
    - "mp"      : Materials Project base model (177 classes)
    - "icsd"    : ICSD transfer model (5203 classes)
    - "cod"     : COD transfer model (2610 classes)
    - "overlap" : MP/COD/ICSD overlap model (84 classes)

Hyper-parameters are shared across all models and are read from
``Model/Checkpoints/basemodel_177.pth``, whose ``best_params`` describes the
XRDNet architecture. Each model's own checkpoint only contributes its weights
(``model_state_dict``).

Run from anywhere; all paths are resolved relative to this file.
"""

import os
import sys
import json
import glob
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Resolve the repository root and add Model/ to sys.path so that the shared
# model definitions in basemodel.py can be imported regardless of the CWD.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(BASE_DIR, "Model")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from basemodel import XRDNet, XRDDataset  


def load_checkpoint_for_test(model: nn.Module, checkpoint_path: str):
    """Load model weights from a raw state_dict or a checkpoint dict."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location="cpu")
    state_dict = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return missing, unexpected


def normalize_single_sample_shape(arr: np.ndarray) -> np.ndarray:
    """Normalize one sample to the expected shape (4501, 1)."""
    if arr.ndim == 3 and arr.shape == (1, 4501, 1):
        return arr[0]
    if arr.ndim == 2 and arr.shape == (4501, 1):
        return arr
    if arr.ndim == 1 and arr.shape[0] == 4501:
        return arr.reshape(4501, 1)
    raise ValueError(f"Unsupported sample shape: {arr.shape}")


def load_rruff_folder_as_dataset(rruff_folder: str, n_phases: int):
    """Load all .npy spectra in a folder into an XRDDataset.

    Labels are dummy one-hot vectors used only to satisfy the XRDDataset
    signature; they are not used during inference.
    """
    npy_files = sorted(glob.glob(os.path.join(rruff_folder, "*.npy")))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in: {rruff_folder}")

    samples = []
    file_names = []
    for path in npy_files:
        arr = np.load(path)
        arr = normalize_single_sample_shape(arr)
        samples.append(arr)
        file_names.append(os.path.basename(path))

    samples = np.array(samples, dtype=np.float32)  # (N, 4501, 1)

    labels = np.zeros((len(samples), n_phases), dtype=np.float32)
    labels[:, 0] = 1.0

    return XRDDataset(samples, labels), file_names


def load_label_to_compound_map(csv_path: str, label_col: str | None = None, compound_col: str | None = None):
    """Read a CSV and return a mapping ``{label_index: compound_name}``.

    When ``label_col`` / ``compound_col`` are not provided, the columns are
    guessed from their names (``label``/``index``/``id`` and
    ``compound``/``chem``/``name`` respectively).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Label mapping CSV not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="gbk")

    if df.shape[1] < 2:
        raise ValueError(f"CSV must have at least two columns: {csv_path}")

    if label_col is None or compound_col is None:
        auto_label_col = None
        auto_compound_col = None
        for col in df.columns:
            lower_col = str(col).lower()
            if auto_label_col is None and ("label" in lower_col or "index" in lower_col or lower_col == "id"):
                auto_label_col = col
            if auto_compound_col is None and ("compound" in lower_col or "chem" in lower_col or "name" in lower_col):
                auto_compound_col = col

        if label_col is None:
            label_col = auto_label_col if auto_label_col is not None else df.columns[0]
        if compound_col is None:
            compound_col = auto_compound_col if auto_compound_col is not None else df.columns[1]

    if label_col not in df.columns:
        raise ValueError(f"Label column not found: {label_col} in {csv_path}")
    if compound_col not in df.columns:
        raise ValueError(f"Compound column not found: {compound_col} in {csv_path}")

    mapping = {}
    for _, row in df.iterrows():
        try:
            label_idx = int(row[label_col])
            comp_name = str(row[compound_col]).strip()
            mapping[label_idx] = comp_name
        except Exception:
            continue

    if not mapping:
        raise ValueError(f"No valid label->compound rows parsed from: {csv_path}")

    return mapping


def parse_elements_from_formula(name: str):
    """Extract element symbols from a formula-like string."""
    # For an overlap class_key such as 'CaTiO3|62', keep only the compound side.
    compound_part = str(name).split("|", 1)[0]
    return set(re.findall(r"[A-Z][a-z]?", compound_part))


def build_hard_mask(label_to_compound, n_classes: int, required_elements, only_allowed_elements=False):
    """Build a bool mask marking the classes allowed by the required elements."""
    required_set = set(required_elements)
    if not required_set:
        return np.ones(n_classes, dtype=bool)

    mask = np.zeros(n_classes, dtype=bool)
    for label_idx in range(n_classes):
        if label_idx not in label_to_compound:
            continue
        comp_name = label_to_compound[label_idx]
        comp_elements = parse_elements_from_formula(comp_name)
        if only_allowed_elements:
            matched = required_set.issubset(comp_elements) and comp_elements.issubset(required_set)
        else:
            matched = required_set.issubset(comp_elements)

        if matched:
            mask[label_idx] = True

    if not mask.any():
        raise ValueError(
            f"Hard mask removed all classes. required_elements={sorted(required_set)}"
        )
    return mask


def build_mp_hard_mask_from_fine_table(csv_path: str, n_classes: int, required_elements, only_allowed_elements=False):
    """Build an MP-specific hard mask from the coarse-cluster summary table.

    1) filter rows by the element prior on column 'Compound';
    2) keep the allowed coarse labels from column 'cluster_id';
    3) use 'Representative_Compound' as the display name for kept labels.
    """
    required_set = set(required_elements)
    if not required_set:
        return np.ones(n_classes, dtype=bool), {}

    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="gbk")

    needed_cols = ["Compound", "cluster_id", "Representative_Compound"]
    for col in needed_cols:
        if col not in df.columns:
            raise ValueError(f"MP hard-mask requires column '{col}' in {csv_path}")

    filtered = []
    for _, row in df.iterrows():
        fine_compound = str(row["Compound"]).strip()
        elems = parse_elements_from_formula(fine_compound)
        if only_allowed_elements:
            matched = required_set.issubset(elems) and elems.issubset(required_set)
        else:
            matched = required_set.issubset(elems)

        if matched:
            filtered.append(row)

    if not filtered:
        raise ValueError(
            f"MP hard mask removed all rows by compound filtering. required_elements={sorted(required_set)}"
        )

    mask = np.zeros(n_classes, dtype=bool)
    label_to_rep = {}
    for row in filtered:
        try:
            coarse_label = int(row["cluster_id"])
        except Exception:
            continue
        if 0 <= coarse_label < n_classes:
            mask[coarse_label] = True
            # Keep the first representative compound for each coarse label.
            if coarse_label not in label_to_rep:
                label_to_rep[coarse_label] = str(row["Representative_Compound"]).strip()

    if not mask.any():
        raise ValueError(
            f"MP hard mask removed all coarse labels. required_elements={sorted(required_set)}"
        )

    return mask, label_to_rep


def apply_hard_mask_to_logits(logits: torch.Tensor, hard_mask_np):
    """Set disallowed class logits to a very small value."""
    if hard_mask_np is None:
        return logits
    hard_mask = torch.tensor(hard_mask_np, device=logits.device, dtype=torch.bool)
    return logits.masked_fill(~hard_mask.unsqueeze(0), -1e9)


def predict_compounds(model, loader, file_names, label_to_compound, device, top_k=5, hard_mask_np=None):
    """Run inference and return top-k predicted compounds per sample.

    Predictions are ranked by softmax probability.
    """
    model.eval()

    results = []
    offset = 0

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            logits = model(x)
            logits = apply_hard_mask_to_logits(logits, hard_mask_np)
            probs = torch.softmax(logits, dim=1)
            k = min(top_k, probs.size(1))
            topk_probs, topk_indices = torch.topk(probs, k=k, dim=1)
            topk_idx = topk_indices.cpu().numpy()
            topk_probs = topk_probs.cpu().numpy()

            for i in range(topk_idx.shape[0]):
                row = {"file_name": file_names[offset + i]}
                for rank in range(k):
                    idx = int(topk_idx[i, rank])
                    row[f"top{rank + 1}_label"] = idx
                    row[f"top{rank + 1}_compound"] = label_to_compound.get(idx, f"UNKNOWN_LABEL_{idx}")
                    row[f"top{rank + 1}_probability"] = float(topk_probs[i, rank])
                results.append(row)
            offset += topk_idx.shape[0]

    return results


def save_prediction_results(results, output_dir):
    """Write prediction results to JSON and CSV files, then return their paths."""
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "predicted_compounds.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(output_dir, "predicted_compounds.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        topk = 0
        if results:
            topk = len([k for k in results[0].keys() if k.startswith("top") and k.endswith("_label")])

        headers = ["file_name"]
        for rank in range(1, topk + 1):
            headers.extend([f"top{rank}_label", f"top{rank}_compound", f"top{rank}_probability"])

        f.write(",".join(headers) + "\n")
        for row in results:
            values = [str(row.get(h, "")).replace(",", " ") for h in headers]
            f.write(",".join(values) + "\n")

    return json_path, csv_path


if __name__ == "__main__":
    # Choose the model to evaluate: "mp", "icsd", "cod", or "overlap".
    DATASET_NAME = "mp"

    # Paths (relative to the repository root).
    RRUFF_FOLDER = os.path.join(SCRIPT_DIR, "RRUFF_processed")
    OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "experiments_report", "real_test")

    # Shared architecture hyper-parameters for every model.
    BASE_PARAMS_PATH = "Model/Checkpoints/basemodel_177.pth"

    MODEL_CONFIGS = {
        "mp": {
            "checkpoint_path": "Model/Checkpoints/basemodel_177.pth",
            "label_map_path": "Clustering/compound_cluster_summary.csv",
            "label_col": "cluster_id",
            "compound_col": "Representative_Compound",
            "n_phases": 177,
        },
        "icsd": {
            "checkpoint_path": "Model/Checkpoints/icsd_best_5203.pth",
            "label_map_path": "comp_label_map/icsd_comp_labels.csv",
            "label_col": "class_index",
            "compound_col": "compound_name",
            "n_phases": 5203,
        },
        "cod": {
            "checkpoint_path": "Model/Checkpoints/cod_best_2610.pth",
            "label_map_path": "comp_label_map/cod_comp_labels.csv",
            "label_col": "class_index",
            "compound_col": "compound_name",
            "n_phases": 2610,
        },
        "overlap": {
            "checkpoint_path": "Model/Checkpoints/best_model_overlap84.pth",
            "label_map_path": "comp_label_map/o84_comp_labels.csv",
            "label_col": "label",
            "compound_col": "class_key",
            "n_phases": 84,
        },
    }

    BATCH_SIZE = 64
    TOP_K = 5

    # Hard-mask config: only classes containing all required elements are kept.
    USE_HARD_MASK = False
    REQUIRED_ELEMENTS = ["Ca", "Ti", "O"]
    ONLY_ALLOWED_ELEMENTS = False  # True: keep only compounds made of exactly these elements

    if DATASET_NAME not in MODEL_CONFIGS:
        raise ValueError(f"Unknown DATASET_NAME={DATASET_NAME}, choose from {list(MODEL_CONFIGS.keys())}")

    cfg = MODEL_CONFIGS[DATASET_NAME]
    CHECKPOINT_PATH = os.path.join(BASE_DIR, cfg["checkpoint_path"])
    LABEL_MAP_PATH = os.path.join(BASE_DIR, cfg["label_map_path"])
    LABEL_COL = cfg["label_col"]
    COMPOUND_COL = cfg["compound_col"]
    N_PHASES = cfg["n_phases"]
    OUTPUT_DIR = os.path.join(OUTPUT_ROOT, DATASET_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading hyper-parameters...")
    params = torch.load(os.path.join(BASE_DIR, BASE_PARAMS_PATH), map_location="cpu")
    best_param = params["best_params"]

    print(f"n_phases = {N_PHASES}")

    print("Loading label-to-compound mapping...")
    label_to_compound = load_label_to_compound_map(
        LABEL_MAP_PATH,
        label_col=LABEL_COL,
        compound_col=COMPOUND_COL,
    )
    print(f"Loaded {len(label_to_compound)} label mappings")

    print("Loading RRUFF npy files...")
    dataset, file_names = load_rruff_folder_as_dataset(RRUFF_FOLDER, n_phases=N_PHASES)
    print(f"Loaded {len(dataset)} files")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = XRDNet(n_phases=N_PHASES, best_params=best_param).to(device)
    missing, unexpected = load_checkpoint_for_test(model, CHECKPOINT_PATH)

    if missing:
        print(f"Warning: missing keys when loading checkpoint: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys when loading checkpoint: {unexpected}")

    print("Predicting compounds on RRUFF_processed...")

    hard_mask_np = None
    if USE_HARD_MASK:
        if DATASET_NAME == "mp":
            hard_mask_np, mp_label_to_rep = build_mp_hard_mask_from_fine_table(
                csv_path=LABEL_MAP_PATH,
                n_classes=N_PHASES,
                required_elements=REQUIRED_ELEMENTS,
                only_allowed_elements=ONLY_ALLOWED_ELEMENTS,
            )
            # Update display names with the filtered representative compounds.
            label_to_compound.update(mp_label_to_rep)
        else:
            hard_mask_np = build_hard_mask(
                label_to_compound,
                n_classes=N_PHASES,
                required_elements=REQUIRED_ELEMENTS,
                only_allowed_elements=ONLY_ALLOWED_ELEMENTS,
            )
        print(
            f"Hard mask enabled. Required elements={REQUIRED_ELEMENTS}, "
            f"only_allowed_elements={ONLY_ALLOWED_ELEMENTS}, "
            f"allowed classes={int(hard_mask_np.sum())}/{N_PHASES}"
        )

    results = predict_compounds(
        model,
        loader,
        file_names,
        label_to_compound,
        device,
        top_k=TOP_K,
        hard_mask_np=hard_mask_np,
    )

    json_path, csv_path = save_prediction_results(results, OUTPUT_DIR)
    print(f"Saved predicted compounds JSON: {json_path}")
    print(f"Saved predicted compounds CSV: {csv_path}")
